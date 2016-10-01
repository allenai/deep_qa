import gzip
import logging
import pickle

from itertools import zip_longest
from typing import List

from overrides import overrides
import numpy

from sklearn.neighbors import LSHForest

from ..data.dataset import TextDataset
from ..data.text_instance import TrueFalseInstance, BackgroundInstance
from .memory_network import MemoryNetworkSolver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DifferentiableSearchSolver(MemoryNetworkSolver):
    """
    A DifferentiableSearchSolver is a MemoryNetworkSolver that does its own search over a corpus to
    find relevant background knowledge for a given input sentence, instead of being reliant on some
    external code (such as Lucene) to do the search for us.

    The only thing we have to change here is re-computing the background info in
    self._pre_epoch_hook(), along with adding a few command-line arguments.

    To do the search, we encode a corpus of sentences using a sentence encoder (e.g., an LSTM),
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
        self.load_saved_lsh = kwargs['load_saved_lsh']

        self.lsh = LSHForest(random_state=12345)
        self.instance_index = {}  # type: Dict[int, str]

    @classmethod
    @overrides
    def update_arg_parser(cls, parser):
        super(DifferentiableSearchSolver, cls).update_arg_parser(parser)
        parser.add_argument('--corpus_path', type=str,
                            help="Location of corpus to use for background knowledge search. "
                            "This corpus is assumed to be gzipped, one sentence per line.")
        parser.add_argument('--load_saved_lsh', action='store_true',
                            help="Only meaningful if you are loading a model.  When loading, "
                            "should we load a pickled LSH, or should we re-initialize the LSH "
                            "from the input corpus?  Note that if you give a corpus path, and you "
                            "load a saved LSH that was constructed from a _different_ corpus, you "
                            "could end up with really weird behavior.")
        parser.add_argument('--num_background', type=int, default=10,
                            help="Number of background sentences to collect for each input")
        parser.add_argument('--num_epochs_delay', type=int, default=10,
                            help="Wait this many epochs before running differentiable search. "
                            "This lets you train with the base memory network code using external "
                            "background knowledge for a time, then, once the encoder is trained "
                            "sufficiently, you can turn on the differentiable search.")
        # TODO(matt): consider only re-encoding at early stopping, instead of a
        # number-of-epoch-based parameter.
        parser.add_argument('--num_epochs_per_encoding', type=int, default=2,
                            help="Number of epochs we wait in between re-encoding the corpus")

    @overrides
    def load_model(self, epoch: int=None):
        """
        Other than loading the model (which we leave to the super class), we also initialize the
        LSH, assuming that if you're loading a model you probably want access to search over the
        input corpus, either for making predictions or for finding nearest neighbors.

        We can either load a saved LSH, or re-initialize it with a new corpus, depending on
        self.load_saved_lsh.
        """
        super(DifferentiableSearchSolver, self).load_model(epoch)
        if self.load_saved_lsh:
            lsh_file = open("%s_lsh.pkl" % self.model_prefix, "rb")
            sentence_index_file = open("%s_index.pkl" % self.model_prefix, "rb")
            self.lsh = pickle.load(lsh_file)
            self.instance_index = pickle.load(sentence_index_file)
            lsh_file.close()
            sentence_index_file.close()
        else:
            self._initialize_lsh()

    def get_nearest_neighbors(self, instance: TrueFalseInstance) -> List[TrueFalseInstance]:
        '''
        Search in the corpus for the nearest neighbors to `instance`.  The corpus we search, how
        many neighbors to return, and the specifics of the encoder model are all defined in
        parameters passed to the constructor of this object.
        '''
        sentence_vector = self.get_sentence_vector(instance.text)
        _, nearest_neighbor_indices = self.lsh.kneighbors([sentence_vector],
                                                          n_neighbors=self.num_background)
        return [self.instance_index[neighbor_index] for neighbor_index in nearest_neighbor_indices[0]]

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
            # First we encode the corpus and (re-)build an LSH.
            self._initialize_lsh()

            # Finally, we update both self.training_dataset and self.validation_dataset with new
            # background information, taken from a nearest neighbor search over the corpus.
            logger.info("Updating the training data background")
            self.training_dataset = self._update_background_dataset(self.training_dataset)
            self.train_input, self.train_labels = self.prep_labeled_data(
                    self.training_dataset, for_train=False, shuffle=True)
            logger.info("Updating the validation data background")
            self.validation_dataset = self._update_background_dataset(self.validation_dataset)
            self.validation_input, self.validation_labels = self._prep_question_dataset(
                    self.validation_dataset)

    @overrides
    def _save_model(self, epoch: int):
        """
        In addition to whatever superclasses do, here we need to save the LSH, so we can load it
        later if desired.
        """
        super(DifferentiableSearchSolver, self)._save_model(epoch)
        lsh_file = open("%s_lsh.pkl" % self.model_prefix, "wb")
        sentence_index_file = open("%s_index.pkl" % self.model_prefix, "wb")
        pickle.dump(self.lsh, lsh_file)
        pickle.dump(self.instance_index, sentence_index_file)
        lsh_file.close()
        sentence_index_file.close()

    def _initialize_lsh(self, batch_size=100):
        """
        This method encodes the corpus in batches, using encoder_model initialized above.  After
        the whole corpus is encoded, we pass the vectors off to sklearn's LSHForest.fit() method.
        """
        if self._sentence_encoder_model is None:
            self._build_sentence_encoder_model()
        logger.info("Reading corpus file")
        corpus_file = gzip.open(self.corpus_path)
        corpus_lines = [line.decode('utf-8') for line in corpus_file.readlines()]

        # Because we're calling as_training_data() on the instances, we need them to have a label,
        # so we pass label=True here.  TODO(matt): make it so that we can get just the input from
        # an instance without the label somehow.
        logger.info("Creating dataset")
        dataset = TextDataset.read_from_lines(corpus_lines,
                                              self._instance_type(),
                                              label=True,
                                              tokenizer=self.tokenizer)
        logger.info("Indexing and padding dataset")
        indexed_dataset = self._index_and_pad_dataset(dataset, self._get_max_lengths())

        def _get_generator():
            instances = zip(dataset.instances, indexed_dataset.instances)
            grouped_instances = zip_longest(*(iter(instances),) * batch_size)
            for batch in grouped_instances:
                batch = [x for x in batch if x is not None]
                yield batch

        generator = _get_generator()
        encoded_sentences = []
        logger.info("Encoding the background corpus")
        num_batches = len(dataset.instances) / batch_size
        log_every = max(1, int(num_batches / 100))
        batch_num = 0
        for batch in generator:
            batch_num += 1
            if batch_num % log_every == 0:
                logger.info("Processing batch %d / %d", batch_num, num_batches)
            instances, indexed_instances = zip(*batch)

            for instance in instances:
                self.instance_index[len(self.instance_index)] = instance

            encoder_input = [instance.as_training_data()[0] for instance in indexed_instances]
            encoder_input = numpy.asarray(encoder_input, dtype='int32')
            current_batch_encoded_sentences = self._sentence_encoder_model.predict(encoder_input)
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
        for instance in dataset.instances:  # type: BackgroundInstance
            text_instance = TrueFalseInstance(instance.text, label=True, tokenizer=self.tokenizer)
            new_background = self.get_nearest_neighbors(text_instance)
            background_text = [background.text for background in new_background]
            new_instances.append(BackgroundInstance(instance, background_text))
        return TextDataset(new_instances)
