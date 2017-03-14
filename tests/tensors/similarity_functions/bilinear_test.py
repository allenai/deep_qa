# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K

from deep_qa.tensors.similarity_functions.bilinear import Bilinear

class TestBilinearSimilarityFunction:
    def test_initialize_weights_returns_correct_weight_sizes(self):
        bilinear = Bilinear(name='bilinear')
        weights = bilinear.initialize_weights(3, 3)
        assert isinstance(weights, list) and len(weights) == 2
        weight_vector, bias = weights
        assert K.int_shape(weight_vector) == (3, 3)
        assert K.int_shape(bias) == (1,)

        weights = bilinear.initialize_weights(2, 5)
        assert isinstance(weights, list) and len(weights) == 2
        weight_vector, bias = weights
        assert K.int_shape(weight_vector) == (2, 5)
        assert K.int_shape(bias) == (1,)

    def test_compute_similarity_does_a_bilinear_product(self):
        bilinear = Bilinear(name='bilinear')
        weights = numpy.asarray([[-.3, .5], [2.0, -1.0]])
        bilinear.weight_matrix = K.variable(weights)
        bilinear.bias = K.variable(numpy.asarray([.1]))
        a_vectors = numpy.asarray([[1, 1], [-1, -1]])
        b_vectors = numpy.asarray([[1, 0], [0, 1]])
        result = K.eval(bilinear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert_almost_equal(result, [1.8, .6])

    def test_compute_similarity_works_with_higher_order_tensors(self):
        bilinear = Bilinear(name='bilinear')
        weights = numpy.random.rand(4, 7)
        bilinear.weight_matrix = K.variable(weights)
        bilinear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.random.rand(5, 4, 3, 6, 4)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        result = K.eval(bilinear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (5, 4, 3, 6)
        expected_result = numpy.dot(numpy.dot(numpy.transpose(a_vectors[3, 2, 1, 3]), weights),
                                    b_vectors[3, 2, 1, 3])
        assert_almost_equal(result[3, 2, 1, 3], expected_result, decimal=5)
