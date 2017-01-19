# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import keras.backend as K

from deep_qa.common import tensors
from .test_markers import requires_tensorflow

class TestTensors:
    def test_masked_batch_dot_masks_properly(self):
        embedding_dim = 3
        a_length = 4
        b_length = 5
        batch_size = 2

        tensor_a = numpy.random.rand(batch_size, a_length, embedding_dim)
        tensor_b = numpy.random.rand(batch_size, b_length, embedding_dim)
        mask_a = numpy.ones((batch_size, a_length))
        mask_a[1, 3] = 0
        mask_b = numpy.ones((batch_size, b_length))
        mask_b[1, 2] = 0
        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 K.variable(mask_a),
                                                 K.variable(mask_b)))
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.all(result[1, 3, :] == numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.all(result[1, :, 2] == numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 None,
                                                 None))
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.all(result[1, :, :] != numpy.zeros((a_length, b_length)))

        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 K.variable(mask_a),
                                                 None))
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.all(result[1, 3, :] == numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 2] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 None,
                                                 K.variable(mask_b)))
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.any(result[1, 0, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 1, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 2, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, 3, :] != numpy.zeros((b_length)))
        assert numpy.any(result[1, :, 0] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 1] != numpy.zeros((a_length)))
        assert numpy.all(result[1, :, 2] == numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 3] != numpy.zeros((a_length)))
        assert numpy.any(result[1, :, 4] != numpy.zeros((a_length)))

    def test_masked_batch_dot_handles_uneven_tensors(self):
        # We're going to test masked_batch_dot with tensors of shape (batch_size, a_length,
        # embedding_dim) and (batch_size, embedding_dim).  The result should have shape
        # (batch_size, a_length).
        embedding_dim = 3
        a_length = 5
        batch_size = 2

        tensor_a = numpy.random.rand(batch_size, a_length, embedding_dim)
        tensor_b = numpy.random.rand(batch_size, embedding_dim)
        mask_a = numpy.ones((batch_size, a_length))
        mask_a[0, 3] = 0
        mask_b = numpy.ones((batch_size,))
        mask_b[1] = 0
        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 K.variable(mask_a),
                                                 K.variable(mask_b)))
        assert result[0, 0] != 0
        assert result[0, 1] != 0
        assert result[0, 2] != 0
        assert result[0, 3] == 0
        assert result[0, 4] != 0
        assert numpy.all(result[1, :] == numpy.zeros((a_length)))

        # We should get the same result if we flip the order of the tensors.
        flipped_result = K.eval(tensors.masked_batch_dot(K.variable(tensor_b),
                                                         K.variable(tensor_a),
                                                         K.variable(mask_b),
                                                         K.variable(mask_a)))
        assert numpy.all(result == flipped_result)

    @requires_tensorflow
    def test_masked_batch_dot_handles_uneven_higher_order_tensors(self):
        # We're going to test masked_batch_dot with tensors of shape (batch_size, common,
        # a_length, embedding_dim) and (batch_size, common, embedding_dim).  The result should have
        # shape (batch_size, common, a_length).  This currently doesn't work with the theano
        # backend, because of an inconsistency in K.batch_dot for higher-order tensors.  The code
        # will crash if you try to run this in Theano, so we require tensorflow for this test.
        embedding_dim = 3
        common_length = 4
        a_length = 5
        batch_size = 2

        tensor_a = numpy.random.rand(batch_size, common_length, a_length, embedding_dim)
        tensor_b = numpy.random.rand(batch_size, common_length, embedding_dim)
        mask_a = numpy.ones((batch_size, common_length, a_length))
        mask_a[1, 1, 3] = 0
        mask_b = numpy.ones((batch_size, common_length))
        mask_b[1, 2] = 0
        result = K.eval(tensors.masked_batch_dot(K.variable(tensor_a),
                                                 K.variable(tensor_b),
                                                 K.variable(mask_a),
                                                 K.variable(mask_b)))
        assert numpy.all(result[0, :, :] != numpy.zeros((common_length, a_length)))
        assert numpy.all(result[1, 0, :] != numpy.zeros((a_length)))
        assert result[1, 1, 0] != 0
        assert result[1, 1, 1] != 0
        assert result[1, 1, 2] != 0
        assert result[1, 1, 3] == 0
        assert result[1, 1, 4] != 0
        assert numpy.all(result[1, 2, :] == numpy.zeros((a_length)))
        assert numpy.all(result[1, 3, :] != numpy.zeros((a_length)))

        # We should get the same result if we pass the smaller tensor in first.
        flipped_result = K.eval(tensors.masked_batch_dot(K.variable(tensor_b),
                                                         K.variable(tensor_a),
                                                         K.variable(mask_b),
                                                         K.variable(mask_a)))
        assert numpy.all(result == flipped_result)

    def test_l1_normalize(self):
        # test 1D case
        vector_1d = K.variable(numpy.array([[2, 1, 5, 7]]))
        vector_1d_normalized = K.eval(tensors.l1_normalize(vector_1d))
        assert_array_almost_equal(vector_1d_normalized,
                                  numpy.array([[0.13333333, 0.06666666,
                                                0.33333333, 0.46666666]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_normalized), decimal=6)
