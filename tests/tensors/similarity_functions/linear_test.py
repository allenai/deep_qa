# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K

from deep_qa.tensors.similarity_functions.linear import Linear

class TestLinearSimilarityFunction:
    def test_initialize_weights_returns_correct_weight_sizes(self):
        linear = Linear(name='linear', combination='x,y')
        weights = linear.initialize_weights(3, 6)
        assert isinstance(weights, list) and len(weights) == 2
        weight_vector, bias = weights
        assert K.int_shape(weight_vector) == (9, 1)
        assert K.int_shape(bias) == (1,)

    def test_compute_similarity_does_a_weighted_product(self):
        linear = Linear(name='linear', combination='x,y')
        linear.weight_vector = K.variable(numpy.asarray([[-.3], [.5], [2.0], [-1.0]]))
        linear.bias = K.variable(numpy.asarray([.1]))
        a_vectors = numpy.asarray([[[1, 1, 1], [-1, -1, 0]]])
        b_vectors = numpy.asarray([[[0], [1]]])
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (1, 2,)
        assert_almost_equal(result, [[2.3, -1.1]])

    def test_compute_similarity_works_with_higher_order_tensors(self):
        linear = Linear(name='linear', combination='x,y')
        weights = numpy.random.rand(14, 1)
        linear.weight_vector = K.variable(weights)
        linear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (5, 4, 3, 6)
        combined_vectors = numpy.concatenate([a_vectors[3, 2, 1, 3, :], b_vectors[3, 2, 1, 3, :]])
        expected_result = numpy.dot(combined_vectors, weights)
        assert_almost_equal(result[3, 2, 1, 3], expected_result, decimal=6)

    def test_compute_similarity_works_with_multiply_combinations(self):
        linear = Linear(name='linear', combination='x*y')
        linear.weight_vector = K.variable(numpy.asarray([[-.3], [.5]]))
        linear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.asarray([[1, 1], [-1, -1]])
        b_vectors = numpy.asarray([[1, 0], [0, 1]])
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert_almost_equal(result, [-.3, -.5])

    def test_compute_similarity_works_with_divide_combinations(self):
        linear = Linear(name='linear', combination='x/y')
        linear.weight_vector = K.variable(numpy.asarray([[-.3], [.5]]))
        linear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.asarray([[1, 1], [-1, -1]])
        b_vectors = numpy.asarray([[1, 2], [2, 1]])
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert_almost_equal(result, [-.05, -.35])

    def test_compute_similarity_works_with_add_combinations(self):
        linear = Linear(name='linear', combination='x+y')
        linear.weight_vector = K.variable(numpy.asarray([[-.3], [.5]]))
        linear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.asarray([[1, 1], [-1, -1]])
        b_vectors = numpy.asarray([[1, 0], [0, 1]])
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert_almost_equal(result, [-.1, .3])

    def test_compute_similarity_works_with_subtract_combinations(self):
        linear = Linear(name='linear', combination='x-y')
        linear.weight_vector = K.variable(numpy.asarray([[-.3], [.5]]))
        linear.bias = K.variable(numpy.asarray([0]))
        a_vectors = numpy.asarray([[1, 1], [-1, -1]])
        b_vectors = numpy.asarray([[1, 0], [0, 1]])
        result = K.eval(linear.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (2,)
        assert_almost_equal(result, [.5, -.7])
