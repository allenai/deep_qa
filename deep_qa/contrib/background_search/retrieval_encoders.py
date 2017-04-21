from collections import OrderedDict
import gzip
import logging
from typing import List

import numpy
from overrides import overrides
import pyhocon
import spacy
import tqdm

from ...common.models import get_submodel
from ...common.params import replace_none, Params
from ...common import util
from ...data.instances.sentence_selection.sentence_selection_instance import SentenceSelectionInstance
from ...models import concrete_models

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RetrievalEncoder:
    """
    An abstract base class for algorithms that encode queries and passages into a vector space, so
    that we can perform vector-based retrieval over the passages given the queries.

    We provide four methods: ``encode_query``, ``encode_passage``, and plural versions of these two
    that handle lists.  Depending on the encoding algorithm, encoding the query and the passage
    might be the same, but we give different methods to allow subclasses to have different
    encodings if they wish.  The default implementation of the plural version of these ``encode``
    methods just calls the singular version in a list comprehension, but a subclass could override
    this to, e.g., make use of batching on a GPU.
    """
    def encode_query(self, query: str) -> numpy.array:
        """
        Converts the query string into a vector.
        """
        raise NotImplementedError

    def encode_passage(self, passage: str) -> numpy.array:
        """
        Converts the passage string into a vector.
        """
        raise NotImplementedError

    def encode_queries(self, queries: List[str]) -> List[numpy.array]:
        """
        Converts the query strings into vectors.
        """
        return [self.encode_query(query) for query in tqdm.tqdm(queries)]

    def encode_passages(self, passages: List[str]) -> List[numpy.array]:
        """
        Converts the passage strings into vectors.
        """
        return [self.encode_passage(passage) for passage in tqdm.tqdm(passages)]


class BagOfWordsRetrievalEncoder(RetrievalEncoder):
    """
    A ``RetrievalEncoder`` that encodes both queries and passages as a bag of pre-trained word
    embeddings.

    We use spacy to tokenize the sentence.

    The ``type`` of this model to use in a parameter file is ``"bow"``.

    Parameters
    ----------
    embeddings_file: str
        A GloVe-formatted gzipped file containing pre-trained word embeddings.

    TODO(matt): I wrote this from an earlier version of ``bow_lsh.py``, before Pradeep implemented
    his IDF feature.  We should update this class to also have an option for IDF encoding, and then
    we can officially retire ``bow_lsh.py``.
    """
    def __init__(self, params: Params):
        embeddings_file = params.pop('embeddings_file')
        self.en_nlp = spacy.load('en')

        # These fields will get set in the call to `read_embeddings_file`.
        self.vector_max = -float("inf")
        self.vector_min = float("inf")
        self.embeddings = {}
        self.embedding_dim = None
        self.read_embeddings_file(embeddings_file)

    def read_embeddings_file(self, embeddings_file: str):
        logger.info("Reading embeddings file: %s", embeddings_file)
        with gzip.open(embeddings_file, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.decode('utf-8').strip().split(' ')
                self.embedding_dim = len(fields) - 1
                word = fields[0]
                vector = numpy.asarray(fields[1:], dtype='float32')
                vector_min = min(vector)
                vector_max = max(vector)
                if vector_min < self.vector_min:
                    self.vector_min = vector_min
                if vector_max > self.vector_max:
                    self.vector_max = vector_max
                self.embeddings[word] = vector

    @overrides
    def encode_query(self, query: str) -> numpy.array:
        return self._encode_sentence(query, for_background=False)

    @overrides
    def encode_passage(self, passage: str) -> numpy.array:
        return self._encode_sentence(passage, for_background=True)

    def _encode_sentence(self, sentence: str, for_background=False):
        words = [str(w.lower_) for w in self.en_nlp.tokenizer(sentence)]
        return numpy.mean(numpy.asarray([self._get_word_vector(word, for_background) for word in words]), axis=0)

    def _get_word_vector(self, word, random_for_unk=False):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            # If this is for the background data, we'd want to make new vectors (uniformly sampling
            # from the range (vector_min, vector_max)). If this is for the queries, we'll return a zero vector
            # for UNK because this word doesn't exist in the background data either.
            if random_for_unk:
                vector = numpy.random.uniform(low=self.vector_min, high=self.vector_max,
                                              size=(self.embedding_dim,))
                self.embeddings[word] = vector
            else:
                vector = numpy.zeros((self.embedding_dim,))
            return vector


class SentenceSelectionRetrievalEncoder(RetrievalEncoder):
    """
    This class takes a trained sentence selection model and uses it to encode passages and queries.

    We make a few assumptions here:

    (1) The sentence selection model must have as its final layer a simple dot product, so that we
        can actually fit the model into this vector-based retrieval paradigm.
    (2) The sentence selection model needs to have its last encoder layers named
        ``question_encoder`` and ``sentences_encoder``.  That is, we'll pull out submodels from the
        sentence selection model using those names, so they need to be present, and should be the
        last thing before doing the similarity computation.  Similarly, the corresponding ``Input``
        layers must have names ``question_input`` and ``sentences_input``, where
        ``sentences_input`` has shape ``(batch_size, num_sentences, sentence_shape)``, and
        ``question_input`` has shape ``(batch_size, sentence_shape)``.

    The ``type`` of this model to use in a parameter file is ``"sentence selection"``.

    Parameters
    ----------
    model_param_file: str
        This is the parameter file used to train the sentence selection model with ``run_model.py``.
    """
    def __init__(self, params: Params):
        model_param_file = params.pop('model_param_file')
        model_params = pyhocon.ConfigFactory.parse_file(model_param_file)
        model_params = replace_none(model_params)
        model_type = params.pop_choice('model_class', concrete_models.keys())
        model_class = concrete_models[model_type]
        self.model = model_class(model_params)
        self.model.load_model()
        # Ok, this is pretty hacky, but calling `self._get_encoder(name)` on a TextTrainer with
        # "use default encoder" as the fallback behavior could give you an encoder that doesn't
        # have the name you expect.
        # pylint: disable=protected-access
        question_encoder_name = self.model._get_encoder(name="question",
                                                        fallback_behavior="use default encoder").name
        self.query_encoder_model = get_submodel(self.model.model,
                                                ['question_input'],
                                                [question_encoder_name],
                                                train_model=False,
                                                name='query_encoder_model')
        self.passage_encoder_model = get_submodel(self.model.model,
                                                  ['sentences_input'],
                                                  ['sentences_encoder'],
                                                  train_model=False,
                                                  name='passage_encoder_model')

    @overrides
    def encode_query(self, query: str) -> numpy.array:
        raise RuntimeError("You shouldn't use this method; use the batched version instead")

    @overrides
    def encode_passage(self, passage: str) -> numpy.array:
        raise RuntimeError("You shouldn't use this method; use the batched version instead")

    @overrides
    def encode_queries(self, queries: List[str]) -> List[numpy.array]:
        query_instances = [SentenceSelectionInstance(query, [], None) for query in queries]
        logger.info("Indexing queries")
        indexed_instances = [instance.to_indexed_instance(self.model.data_indexer)
                             for instance in tqdm.tqdm(query_instances)]
        logger.info("Padding queries")
        for instance in tqdm.tqdm(indexed_instances):
            instance.pad(self.model._get_max_lengths())  # pylint: disable=protected-access
        query_arrays = numpy.asarray([instance.as_training_data()[0][0] for instance in indexed_instances])
        logger.info("Getting query vectors")
        return self.query_encoder_model.predict(query_arrays)

    @overrides
    def encode_passages(self, passages: List[str]) -> List[numpy.array]:
        grouped_passages = util.group_by_count(passages, self.model.num_sentences, '')
        passage_instances = [SentenceSelectionInstance('', passage_group, None)
                             for passage_group in grouped_passages]
        logger.info("Indexing passages")
        indexed_instances = [instance.to_indexed_instance(self.model.data_indexer)
                             for instance in tqdm.tqdm(passage_instances)]
        logger.info("Padding passages")
        for instance in tqdm.tqdm(indexed_instances):
            instance.pad(self.model._get_max_lengths())  # pylint: disable=protected-access
        grouped_passage_arrays = numpy.asarray([instance.as_training_data()[0][1]
                                                for instance in indexed_instances])
        logger.info("Getting passage vectors")
        grouped_passage_vectors = self.passage_encoder_model.predict(grouped_passage_arrays)
        shape = grouped_passage_vectors.shape
        new_shape = (shape[0] * shape[1], shape[2])
        passage_vectors = grouped_passage_vectors.reshape(new_shape)
        return passage_vectors[:len(passages)]


retrieval_encoders = OrderedDict()  # pylint:  disable=invalid-name
retrieval_encoders['bow'] = BagOfWordsRetrievalEncoder
retrieval_encoders['sentence selection'] = SentenceSelectionRetrievalEncoder
