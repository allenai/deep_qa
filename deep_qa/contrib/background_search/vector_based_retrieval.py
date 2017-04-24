import gzip
import logging
from typing import List, Tuple

import numpy

from ...common.params import Params
from .nearest_neighbor_algorithms import nearest_neighbor_algorithms
from .retrieval_encoders import retrieval_encoders

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class VectorBasedRetrieval:
    """
    This class performs retrieval over a background corpus using vectors to represent both the
    query and the items in the corpus.  This class just gives basic functionality around retrieval;
    the actual details of both how you do lookups and how you encode the query and the background
    corpus are left to other classes, which we contain.

    Parameters
    ----------
    serialization_prefix: str, optional (default='retrieval')
        When we save and load models (both encoders and nearest neighbor indices), we will do so by
        appending things to this path.
    encoder: Dict[str, Any], optional (default={})
        These parameters get passed to the encoder model.  See the specific encoder model for
        options here.  The one parameter looked at in this class is ``type``, which determines the
        actual encoder model to be used.  The rest of the parameters get passed along.
    nearest_neighbors: Dict[str, Any], optional (default={})
        These parameters get passed to the approximate nearest neighbor algorithm.  See the
        specific class for options here.  The one parameter looked at in this class is ``type``,
        which determines the actual algorithm to be used.  The rest of the parameters get passed
        along.

    Notes
    -----
    To use this class, you first need to read some background sentences with
    ``retrieval.read_background``.  Once you've read all background files, you call
    ``retrieval.fit()``, to encode all of the sentences and load them into some approximate nearest
    neighbor algorithm.  Then you can retrieve background sentences given a query string with
    ``retrieval.get_nearest_neighbors``.
    """
    def __init__(self, params: Params):
        self.serialization_prefix = params.pop('serialization_prefix', 'retrieval')

        encoder_params = params.pop('encoder', {})
        encoder_choice = encoder_params.pop_choice('type', list(retrieval_encoders.keys()),
                                                   default_to_first_choice=True)
        self.encoder = retrieval_encoders[encoder_choice](encoder_params)

        nearest_neighbors_params = params.pop('nearest_neighbors', {})
        nearest_neighbors_choice = \
            nearest_neighbors_params.pop_choice('type', list(nearest_neighbor_algorithms.keys()),
                                                default_to_first_choice=True)
        self.nearest_neighbors = nearest_neighbor_algorithms[nearest_neighbors_choice](nearest_neighbors_params)

        params.assert_empty("VectorBasedRetrieval")

        self.background_sentences = []

    def load_model(self):  # pylint: disable=no-self-use
        # TODO(matt): not sure what this should look like yet
        logger.warning("LOADING MODELS NOT IMPLEMENTED YET")

    def save_model(self):  # pylint: disable=no-self-use
        # TODO(matt): not sure what this should look like yet
        logger.warning("SAVING MODELS NOT IMPLEMENTED YET")

    def read_background(self, background_file):
        """
        Reads the given background file, which is assumed to be gzipped, with one retrievable
        passage per line.  All non-empty lines in the file are added to the retrieval index.
        """
        logger.info("Reading background file: %s", background_file)
        # Read background file and add to `background_sentences`.
        for sentence in gzip.open(background_file, mode="r"):
            sentence = sentence.decode('utf-8').strip()
            if sentence != '':
                self.background_sentences.append(sentence)

    def fit(self):
        """
        Encodes all of the read background passages as vectors, using ``self.encoder``, then fits
        them into some approximate nearest neighbor algorithm, such as an LSH.
        """
        logger.info("Fitting nearest neighbor algorithm")
        encoded_passages = self.encoder.encode_passages(self.background_sentences)
        self.nearest_neighbors.fit(encoded_passages)

    def get_nearest_neighbors(self, text_query: str, num_neighbors: int) -> List[Tuple[str, float]]:
        """
        Returns the ``num_neighbors`` closest approximate nearest neighbors to ``text_query`` in
        the background corpus, along with a score for each neighbor.  ``text_query`` can be either
        a single string, or a list of strings.
        """
        if not isinstance(text_query, list):
            text_query = [text_query]
        query_vectors = numpy.asarray(self.encoder.encode_queries(text_query))
        results = self.nearest_neighbors.get_neighbors(query_vectors, num_neighbors)
        results = [self._get_sentences(result) for result in results]
        if len(results) == 1:
            results = results[0]
        return results

    def _get_sentences(self, results: List[Tuple[int, float]]) -> List[Tuple[str, float]]:
        """
        Gets the sentences corresponding to indices in ``results`` (the first field) from
        ``self.background_sentences``.
        """
        return [(self.background_sentences[i], score) for (i, score) in results]
