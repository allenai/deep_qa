import gzip
import logging
import pickle

from itertools import zip_longest
from typing import List

from overrides import overrides
import numpy

from sklearn.neighbors import LSHForest
from keras.models import Model

from ..data.dataset import TextDataset
from ..data.instance import TextInstance, BackgroundTextInstance
from .memory_network import MemoryNetworkSolver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DifferentiableSearchSolver(MemoryNetworkSolver):
    """
    A DifferentiableSearchSolver is a MemoryNetworkSolver that does its own search over a corpus to
    find relevant background knowledge for a given input sentence, instead of being reliant on some
    external code (such as Lucene) to do the search for us.

    The only thing we have to change here is re-computing the background info in
    self.pre_epoch_hook(), along with adding a few command-line arguments.

    To do the search, we encode a corpus of sentences using a sentence encoder (e.g., as LSTM),
    then use nearest-neighbor search on the sentence encodings.

    We perform the nearest-neighbor search using a scikit-learn's locality sensitive hash (LSH).
    All variable names involving "lsh" in this class refer to "locality sensitive hash".

    Note that as this is currently implemented, we will take initial background sentences from a
    file, using the standard MemoryNetworkSolver code.  It is only in subsequent epochs that we
    will override that and use our differentiable search to find background knowledge.
    """
    def __init__(self, **kwargs):
        super(DifferentiableSearchSolver, self).__init__(**kwargs)
        self.corpus_path = kwargs['corpus_path']
        self.num_background = kwargs['num_background']
        self.num_epochs_delay = kwargs['num_epochs_delay']
        self.num_epochs_per_encoding = kwargs['num_epochs_per_encoding']

        self.encoder_model = None
        self.lsh = LSHForest(random_state=12345)
        self.instance_index = {}  # type: Dict[int, str]

    @classmethod
    @overrides
    def update_arg_parser(cls, parser):
        super(DifferentiableSearchSolver, cls).update_arg_parser(parser)
        parser.add_argument('--corpus_path', type=str,
                            help="Location of corpus to use for background knowledge search. "
                            "This corpus is assumed to be gzipped, one sentence per line.")
        parser.add_argument('--num_background', type=int, default=10,
                            help="Number of background sentences to collect for each input")
        parser.add_argument('--num_epochs_delay', type=int, default=10,
                            help="Wait this many epochs before running differentiable search. "
                            "This lets you train with the base memory network code using external "
                            "background knowledge for a time, then, once the encoder is trained "
                            "sufficiently, you can turn on the differentiable search.")
        parser.add_argument('--num_epochs_per_encoding', type=int, default=2,
                            help="Number of epochs we wait in between re-encoding the corpus")

    @overrides
    def load_model(self, epoch: int=None):
        """
        Two things to do here other than what superclasses do: (1) initialize the encoder model,
        and (2) load the LSH (note that this includes the background corpus; if you want to use a
        different corpus at test time, this method has to change).
        """
        super(DifferentiableSearchSolver, self).load_model(epoch)
        self._load_encoder()
        lsh_file = open("%s_lsh.pkl" % self.model_prefix, "rb")
        sentence_index_file = open("%s_index.pkl" % self.model_prefix, "rb")
        self.lsh = pickle.load(lsh_file)
        self.instance_index = pickle.load(sentence_index_file)
        lsh_file.close()
        sentence_index_file.close()

    def get_nearest_neighbors(self, instance: TextInstance) -> List[TextInstance]:
        '''
        Search in the corpus for the nearest neighbors to `instance`.  The corpus we search, how
        many neighbors to return, and the specifics of the encoder model are all defined in
        parameters passed to the constructor of this object.
        '''
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        max_lengths = [self.max_sentence_length, self.max_knowledge_length]
        indexed_instance.pad(max_lengths)
        instance_input, _ = indexed_instance.as_training_data()
        encoded_instance = self.encoder_model.predict(numpy.asarray(instance_input))
        _, nearest_neighbor_indices = self.lsh.kneighbors(
                [encoded_instance], n_neighbors=self.num_background)
        return [self.instance_index[neighbor_index] for neighbor_index in nearest_neighbor_indices]

    @overrides
    def _pre_epoch_hook(self, epoch: int):
        """
        Here is where we re-encode the background knowledge and re-do the nearest neighbor search,
        recreating the train and validation datasets with new background data.

        We wait self.num_epochs_delay before starting this, and then we do it every
        self.num_epochs_per_encoding epochs.
        """
        super(DifferentiableSearchSolver, self)._pre_epoch_hook(epoch)
        if epoch >= self.num_epochs_delay and epoch % self.num_epochs_per_encoding == 0:
            # First we rebuild the encoder model.  TODO(matt): will this actually set weights from
            # the existing layers?  I assume it will.  Pradeep, do you know?
            self._load_encoder()

            # Then we encode the corpus and build an LSH.
            self._initialize_lsh()

            # Finally, we update both self.training_dataset and self.validation_dataset with new
            # background information, taken from a nearest neighbor search over the corpus.
            self.training_dataset = self._update_background_dataset(self.training_dataset)
            self.train_input, self.train_labels = self.prep_labeled_data(self.training_dataset)
            self.validation_dataset = self._update_background_dataset(self.validation_dataset)
            self.validation_input, self.validation_labels = self._prep_question_dataset(
                    self.validation_dataset)

    @overrides
    def _save_model(self, epoch: int):
        """
        In addition to whatever superclasses do, here we need to save the LSH, so we can load it
        later.  Alternatively, we could just reinitialize it on load_model()...
        """
        super(DifferentiableSearchSolver, self)._save_model(epoch)
        lsh_file = open("%s_lsh.pkl" % self.model_prefix, "wb")
        sentence_index_file = open("%s_index.pkl" % self.model_prefix, "wb")
        pickle.dump(self.lsh, lsh_file)
        pickle.dump(self.instance_index, sentence_index_file)
        lsh_file.close()
        sentence_index_file.close()

    def _load_encoder(self):
        """
        Here we pull out just a couple of layers from self.model and use them to define a
        stand-alone encoder model.

        Specifically, we need the part of the model that gets us from word index sequences to word
        embedding sequences, and the part of the model that gets us from word embedding sequences
        to sentence vectors.

        This must be called after self.max_sentence_length has been set, which happens when
        self._get_training_data() is called.
        """
        input_layer, embedded_input = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,))
        encoder_layer = self._get_sentence_encoder()
        encoded_input = encoder_layer(embedded_input)
        self.encoder_model = Model(input=input_layer, output=encoded_input)

        # Loss and optimizer do not matter here since we're not going to train this model. But it
        # needs to be compiled to use it for prediction.
        self.encoder_model.compile(loss="mse", optimizer="adam")

    def _initialize_lsh(self, batch_size=100):
        """
        This method encodes the corpus in batches, using encoder_model initialized above.  After
        the whole corpus is encoded, we pass the vectors off to sklearn's LSHForest.fit() method.
        """
        logger.info("Reading corpus file")
        corpus_file = gzip.open(self.corpus_path)
        corpus_lines = [line.decode('utf-8') for line in corpus_file.readlines()]

        # Because we're calling as_training_data() on the instances, we need them to have a label,
        # so we pass label=True here.  TODO(matt): make it so that we can get just the input from
        # an instance without the label somehow.
        logger.info("Creating dataset")
        dataset = TextDataset.read_from_lines(corpus_lines, label=True)
        max_lengths = [self.max_sentence_length, self.max_knowledge_length]
        logger.info("Indexing and padding dataset")
        indexed_dataset = self._index_and_pad_dataset(dataset, max_lengths)

        def _get_generator():
            instances = zip(dataset.instances, indexed_dataset.instances)
            grouped_instances = zip_longest(*(iter(instances),) * batch_size)
            for batch in grouped_instances:
                batch = [x for x in batch if x is not None]
                yield batch

        generator = _get_generator()
        encoded_sentences = []
        logger.info("Encoding the background corpus")
        for batch in generator:
            instances, indexed_instances = zip(*batch)

            for instance in instances:
                self.instance_index[len(self.instance_index)] = instance

            encoder_input = [instance.as_training_data()[0] for instance in indexed_instances]
            encoder_input = numpy.asarray(encoder_input, dtype='int32')
            current_batch_encoded_sentences = self.encoder_model.predict(encoder_input)
            for encoded_sentence in current_batch_encoded_sentences:
                encoded_sentences.append(encoded_sentence)
        encoded_sentences = numpy.asarray(encoded_sentences)
        logger.info("Fitting the LSH")
        self.lsh.fit(encoded_sentences)

    def _update_background_dataset(self, dataset: TextDataset) -> TextDataset:
        """
        Given a dataset, this method takes all of the instance in that dataset and finds new
        background knowledge for them, returning a new dataset with updated background knowledge.
        """
        new_instances = []
        for instance in dataset.instances:  # type: BackgroundTextInstance
            new_background = self.get_nearest_neighbors(instance)
            background_text = [background.text for background in new_background]
            new_instances.append(BackgroundTextInstance(instance.text,
                                                        background_text,
                                                        instance.label,
                                                        instance.index))
        return TextDataset(new_instances)
