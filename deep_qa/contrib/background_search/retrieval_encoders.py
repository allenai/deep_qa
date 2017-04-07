from collections import OrderedDict
import gzip
import logging
from typing import Any, Dict, List

import numpy
from overrides import overrides
import spacy
import tqdm

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

    Parameters
    ----------
    embeddings_file: str
        A GloVe-formatted gzipped file containing pre-trained word embeddings.

    TODO(matt): I wrote this from an earlier version of ``bow_lsh.py``, before Pradeep implemented
    his IDF feature.  We should update this class to also have an option for IDF encoding, and then
    we can officially retire ``bow_lsh.py``.
    """
    def __init__(self, params: Dict[str, Any]):
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


retrieval_encoders = OrderedDict()  # pylint:  disable=invalid-name
retrieval_encoders['bow'] = BagOfWordsRetrievalEncoder
