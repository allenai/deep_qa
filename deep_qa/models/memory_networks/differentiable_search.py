import gzip
import logging
import pickle

from itertools import zip_longest
from typing import List

from overrides import overrides
import numpy

from sklearn.neighbors import LSHForest

from ...data.dataset import TextDataset
from ...data.instances.wrappers import BackgroundInstance
from ...data.instances.text_classification.text_classification_instance import TextClassificationInstance
from ...common.params import Params
from .memory_network import MemoryNetwork

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DifferentiableSearchMemoryNetwork(MemoryNetwork):
    """
    A DifferentiableSearchMemoryNetwork is a MemoryNetwork that does its own search over a corpus to
    find relevant background knowledge for a given input sentence, instead of being reliant on some
    external code (such as Lucene) to do the search for us.

    The only thing we have to change here is re-computing the background info in
    self._pre_epoch_hook(), along with adding a few command-line arguments.

    To do the search, we encode a corpus of sentences using a sentence encoder (e.g., an LSTM),
    then use nearest-neighbor search on the sentence encodings.

    We perform the nearest-neighbor search using a scikit-learn's locality sensitive hash (LSH).
    All variable names involving "lsh" in this class refer to "locality sensitive hash".

    Note that as this is currently implemented, we will take initial background sentences from a
    file, using the standard MemoryNetwork code.  It is only in subsequent epochs that we
    will override that and use our differentiable search to find background knowledge.
    """
    def __init__(self, params: Params):
        # Location of corpus to use for background knowledge search. This corpus is assumed to be
        # gzipped, one sentence per line.
        self.corpus_path = params.pop('corpus_path', None)

        # Number of background sentences to collect for each input.
        self.num_background = params.pop('num_background', 10)
        # Wait this many epochs before running differentiable search. This lets you train with the
        # base memory network code using external background knowledge for a time, then, once the
        # encoder is trained sufficiently, you can turn on the differentiable search.
        self.num_epochs_delay = params.pop('num_epochs_delay', 10)

        # Number of epochs we wait in between re-encoding the corpus.
        # TODO(matt): consider only re-encoding at early stopping, instead of a
        # number-of-epoch-based parameter.
        self.num_epochs_per_encoding = params.pop('num_epochs_per_encoding', 2)

        # Only meaningful if you are loading a model.  When loading, should we load a pickled LSH,
        # or should we re-initialize the LSH from the input corpus?  Note that if you give a corpus
        # path, and you load a saved LSH that was constructed from a _different_ corpus, you could
        # end up with really weird behavior.
        self.load_saved_lsh = params.pop('load_saved_lsh', False)

        # Now that we've popped our parameters, we can call the superclass constructor.
        super(DifferentiableSearchMemoryNetwork, self).__init__(params)

        # And then set some member variables.
        self._sentence_encoder_model = self.__build_sentence_encoder_model()
        self.lsh = LSHForest(random_state=12345)
        self.instance_index = {}  # type: Dict[int, str]

    @overrides
    def load_model(self, epoch: int=None):
        """
        Other than loading the model (which we leave to the super class), we also initialize the
        LSH, assuming that if you're loading a model you probably want access to search over the
        input corpus, either for making predictions or for finding nearest neighbors.

        We can either load a saved LSH, or re-initialize it with a new corpus, depending on
        self.load_saved_lsh.
        """
        super(DifferentiableSearchMemoryNetwork, self).load_model(epoch)
        if self.load_saved_lsh:
            lsh_file = open("%s_lsh.pkl" % self.model_prefix, "rb")
            sentence_index_file = open("%s_index.pkl" % self.model_prefix, "rb")
            self.lsh = pickle.load(lsh_file)
            self.instance_index = pickle.load(sentence_index_file)
            lsh_file.close()
            sentence_index_file.close()
        else:
            self._initialize_lsh()

    def get_nearest_neighbors(self, instance: TextClassificationInstance) -> List[TextClassificationInstance]:
        '''
        Search in the corpus for the nearest neighbors to `instance`.  The corpus we search, how
        many neighbors to return, and the specifics of the encoder model are all defined in
        parameters passed to the constructor of this object.
        '''
        sentence_vector = self.__get_sentence_vector(instance.text)
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
        super(DifferentiableSearchMemoryNetwork, self)._pre_epoch_hook(epoch)
        if epoch >= self.num_epochs_delay and epoch % self.num_epochs_per_encoding == 0:
            # First we encode the corpus and (re-)build an LSH.
            self._initialize_lsh()

            # Then we update both self.training_dataset and self.validation_dataset with new
            # background information, taken from a nearest neighbor search over the corpus.
            logger.info("Updating the training data background")
            self.training_dataset = self._update_background_dataset(self.training_dataset)
            indexed_dataset = self.training_dataset.to_indexed_dataset(self.data_indexer)
            self.training_arrays = self.create_data_arrays(indexed_dataset)
            if self.validation_dataset:
                logger.info("Updating the validation data background")
                self.validation_dataset = self._update_background_dataset(self.validation_dataset)
                indexed_dataset = self.validation_dataset.to_indexed_dataset(self.data_indexer)
                self.validation_arrays = self.create_data_arrays(indexed_dataset)

    @overrides
    def _save_auxiliary_files(self):
        """
        In addition to whatever superclasses do, here we need to save the LSH, so we can load it
        later if desired.
        """
        super(DifferentiableSearchMemoryNetwork, self)._save_auxiliary_files()
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
        logger.info("Reading corpus file")
        corpus_file = gzip.open(self.corpus_path)
        corpus_lines = [line.decode('utf-8') for line in corpus_file.readlines()]

        logger.info("Creating dataset")
        dataset = TextDataset.read_from_lines(corpus_lines, self._instance_type())

        def _get_generator():
            grouped_instances = zip_longest(*(iter(dataset.instances),) * batch_size)
            for batch in grouped_instances:
                batch = [x for x in batch if x is not None]
                yield batch

        generator = _get_generator()
        encoded_sentences = []
        logger.info("Encoding the background corpus")
        num_batches = len(dataset.instances) / batch_size
        log_every = max(1, int(num_batches / 100))
        batch_num = 0
        for instances in generator:
            batch_num += 1
            if batch_num % log_every == 0:
                logger.info("Processing batch %d / %d", batch_num, num_batches)

            for instance in instances:
                self.instance_index[len(self.instance_index)] = instance

            dataset = TextDataset(instances)
            indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
            encoder_input, _ = self.create_data_arrays(indexed_dataset)
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
            text_instance = TextClassificationInstance(instance.text, label=True)
            new_background = self.get_nearest_neighbors(text_instance)
            background_text = [background.text for background in new_background]
            new_instances.append(BackgroundInstance(instance, background_text))
        return TextDataset(new_instances)

    def __build_sentence_encoder_model(self):  # pylint: disable=no-self-use
        # TODO(matt): this should be done using common.models.get_submodel().  I removed the method
        # this used to use from TextTrainer, because it's too specific a need to belong there.
        class Dummy:
            def predict(self, dummy_input):  # pylint: disable=no-self-use,unused-argument
                return numpy.asarray([1])
        return Dummy()

    def __get_sentence_vector(self, sentence: str):  # pylint: disable=no-self-use,unused-argument
        # TODO(matt): this should be done using self._sentence_encoder_model  I removed the method
        # this used to use from TextTrainer, because it's too specific a need to belong there.
        # Also, this whole class should probably just use the vector-based retrieval code.  This
        # class was written pretty early in the life of this codebase, before we had learned a lot
        # of good lessons about writing keras code.
        return numpy.asarray([1])
