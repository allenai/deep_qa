from collections import OrderedDict
import logging
from typing import List, Tuple

import numpy
from sklearn.neighbors import LSHForest
from ...common.params import Params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NearestNeighborAlgorithm:
    """
    An abstract base class for algorithms that perform an approximate nearest neighbor search over
    a large vector space.
    """
    def fit(self, vectors: List[numpy.array]):
        """
        Given a list of vectors, do whatever the algorithm needs to do in order to retrieve
        nearest neighbors from this list efficiently.  When calling ``get_neighbors``, we return
        indices into this list of vectors, so the caller must maintain the list in order to recover
        the vectors (or whatever underlying thing those vectors map to).
        """
        raise NotImplementedError

    def get_neighbors(self, query_vector: numpy.array, num_neighbors: int) -> List[Tuple[int, float]]:
        """
        Returns the ``num_neighbors`` (approximate) nearest neighbors to the ``query_vector`` in
        the vectors that were passed to ``self.fit()``, along with an associate score for each
        neighbor.  This method returns `indices` into the list of vectors passed to ``fit()``, not
        the vectors themselves, because we presume the caller has maintained a mapping from the
        indices to whatever underlying data structure they care about.

        ``query_vector`` can be either a single vector (a 1-dimensional numpy array), or a list of
        vectors (or a 2-dimensional numpy array).
        """
        raise NotImplementedError


class ScikitLearnLsh(NearestNeighborAlgorithm):
    """
    This ``NearestNeighborAlgorithm`` uses scikit-learn's implementation of a locality sensitive
    hash to find approximate nearest neighbors.

    Parameters
    ----------
    random_state: int, optional (default=12345)
        Used to initialize the LSHForest, so that runs are consistent.
    """
    def __init__(self, params: Params):
        random_state = params.pop('random_state', 12345)
        self.lsh = LSHForest(random_state=random_state)

    def fit(self, vectors: List[numpy.array]):
        logger.info("Fitting LSH with %d vectors", len(vectors))
        self.lsh.fit(vectors)

    def get_neighbors(self, query_vector: numpy.array, num_neighbors: int) -> List[Tuple[int, float]]:
        if len(query_vector.shape) == 1:
            query_vector = [query_vector]
        logger.info("Getting neighbors for %d vectors", len(query_vector))
        scores, neighbor_indices = self.lsh.kneighbors(query_vector, n_neighbors=num_neighbors)
        logger.info("Neighbors retrieved")
        result = [zip(neighbor_indices[i], scores[i]) for i in range(len(neighbor_indices))]
        if len(result) == 1:
            result = result[0]
        return result


nearest_neighbor_algorithms = OrderedDict()  # pylint: disable=invalid-name
nearest_neighbor_algorithms['lsh'] = ScikitLearnLsh
