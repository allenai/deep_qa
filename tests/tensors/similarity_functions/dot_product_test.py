# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K

from deep_qa.tensors.similarity_functions.dot_product import DotProduct

class TestDotProductSimilarityFunction:
    dot_product = DotProduct(name='dot_product')
    def test_initialize_weights_returns_empty(self):
        weights = self.dot_product.initialize_weights(input_shape=(2, 3))
        assert isinstance(weights, list) and len(weights) == 0

    def test_compute_similarity_does_a_dot_product(self):
        a_vectors = numpy.asarray([[1, 1, 1], [-1, -1, -1]])
        b_vectors = numpy.asarray([[1, 0, 1], [1, 0, 0]])
        result = K.eval(self.dot_product.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert numpy.all(result == [2, -1])

    def test_compute_similarity_works_with_higher_order_tensors(self):
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        result = K.eval(self.dot_product.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (5, 4, 3, 6)
        assert_almost_equal(result[3, 2, 1, 3],
                            numpy.dot(a_vectors[3, 2, 1, 3], b_vectors[3, 2, 1, 3]),
                            decimal=6)
