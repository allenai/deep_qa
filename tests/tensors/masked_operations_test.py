# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal, assert_array_almost_equal
import keras.backend as K

from deep_qa.tensors.backend import l1_normalize
from deep_qa.tensors.masked_operations import masked_batch_dot, masked_softmax


class TestMaskedOperations:
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
        result = K.eval(masked_batch_dot(K.variable(tensor_a),
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

        result = K.eval(masked_batch_dot(K.variable(tensor_a),
                                         K.variable(tensor_b),
                                         None,
                                         None))
        assert numpy.all(result[0, :, :] != numpy.zeros((a_length, b_length)))
        assert numpy.all(result[1, :, :] != numpy.zeros((a_length, b_length)))

        result = K.eval(masked_batch_dot(K.variable(tensor_a),
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

        result = K.eval(masked_batch_dot(K.variable(tensor_a),
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
        result = K.eval(masked_batch_dot(K.variable(tensor_a),
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
        flipped_result = K.eval(masked_batch_dot(K.variable(tensor_b),
                                                 K.variable(tensor_a),
                                                 K.variable(mask_b),
                                                 K.variable(mask_a)))
        assert numpy.all(result == flipped_result)

    def test_masked_batch_dot_handles_uneven_higher_order_tensors(self):
        # We're going to test masked_batch_dot with tensors of shape (batch_size, common,
        # a_length, embedding_dim) and (batch_size, common, embedding_dim).  The result should have
        # shape (batch_size, common, a_length).
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
        result = K.eval(masked_batch_dot(K.variable(tensor_a),
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
        flipped_result = K.eval(masked_batch_dot(K.variable(tensor_b),
                                                 K.variable(tensor_a),
                                                 K.variable(mask_b),
                                                 K.variable(mask_a)))
        assert numpy.all(result == flipped_result)

    def test_l1_normalize_no_mask(self):
        # Testing the general unmasked 1D case.
        vector_1d = K.variable(numpy.array([[2, 1, 5, 7]]))
        vector_1d_normalized = K.eval(l1_normalize(vector_1d))
        assert_almost_equal(vector_1d_normalized,
                            numpy.array([[0.13333333, 0.06666666,
                                          0.33333333, 0.46666666]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_normalized), decimal=6)

        # Testing the unmasked 1D case with all 0s.
        vector_1d_zeros = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_normalized = K.eval(l1_normalize(vector_1d_zeros))
        assert_array_almost_equal(vector_1d_zeros_normalized,
                                  numpy.array([[0.25, 0.25, 0.25, 0.25]]))

        # Testing the general unmasked batched case when
        # inputs are not all 0's
        matrix = K.variable(numpy.array([[2, 1, 5, 7], [2, 2, 2, 2]]))
        matrix_normalized = K.eval(l1_normalize(matrix))
        assert_array_almost_equal(matrix_normalized,
                                  numpy.array([[0.13333333, 0.06666666,
                                                0.33333333, 0.46666666],
                                               [0.25, 0.25,
                                                0.25, 0.25]]))
        assert_almost_equal(numpy.array([1.0, 1.0]),
                            numpy.sum(matrix_normalized, axis=1), decimal=6)

        # Testing the general unmasked batched case when
        # one row is all 0's
        matrix = K.variable(numpy.array([[2, 1, 5, 7], [0, 0, 0, 0]]))
        matrix_normalized = K.eval(l1_normalize(matrix))
        assert_array_almost_equal(matrix_normalized,
                                  numpy.array([[0.13333333, 0.06666666,
                                                0.33333333, 0.46666666],
                                               [0.25, 0.25,
                                                0.25, 0.25]]))
        assert_almost_equal(numpy.array([1.0, 1.0]),
                            numpy.sum(matrix_normalized, axis=1), decimal=6)

    def test_l1_normalize_masked(self):
        # Testing the general masked 1D case.
        vector_1d = K.variable(numpy.array([[2, 1, 5, 7]]))
        vector_1d_mask = K.variable(numpy.array([[1, 1, 0, 1]]))
        vector_1d_normalized = K.eval(l1_normalize(vector_1d,
                                                   vector_1d_mask))
        assert_array_almost_equal(vector_1d_normalized,
                                  numpy.array([[0.2, 0.1,
                                                0.0, 0.7]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_normalized), decimal=6)

        vector_1d = K.variable(numpy.array([[1.0, 2.0, 3.0, 4.0]]))
        vector_1d_mask = K.variable(numpy.array([[1, 1, 0, 1]]))
        vector_1d_normalized = K.eval(l1_normalize(vector_1d,
                                                   vector_1d_mask))
        assert_array_almost_equal(vector_1d_normalized,
                                  numpy.array([[0.14285715, 0.2857143,
                                                0, 0.5714286]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_normalized), decimal=6)


        # Testing the masked 1D case where the mask is
        # not all zero and the input is all zero.
        vector_1d_zeros = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_mask = K.variable(numpy.array([[1, 1, 0, 1]]))
        vector_1d_zeros_normalized = K.eval(l1_normalize(vector_1d_zeros,
                                                         vector_1d_zeros_mask))
        assert_array_almost_equal(vector_1d_zeros_normalized,
                                  numpy.array([[0.3333333, 0.3333333,
                                                0.0, 0.3333333]]))

        vector_1d_zeros = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_mask = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_normalized = K.eval(l1_normalize(vector_1d_zeros,
                                                         vector_1d_zeros_mask))
        assert_array_almost_equal(vector_1d_zeros_normalized,
                                  numpy.array([[0.25, 0.25,
                                                0.25, 0.25]]))

        # Testing the general batched masked case when the input is not
        # all 0's and the masks are not all 0's.
        matrix = K.variable(numpy.array([[2, 1, 5, 7], [2, 2, 2, 2]]))
        matrix_mask = K.variable(numpy.array([[1, 1, 0, 1], [1, 1, 1, 1]]))
        matrix_normalized = K.eval(l1_normalize(matrix, matrix_mask))
        assert_array_almost_equal(matrix_normalized,
                                  numpy.array([[0.2, 0.1,
                                                0.0, 0.7],
                                               [0.25, 0.25,
                                                0.25, 0.25]]))
        assert_almost_equal(numpy.array([1.0, 1.0]),
                            numpy.sum(matrix_normalized, axis=1), decimal=6)

        # Testing the batched masked case when the masks are all 0's
        # and one of the input rows is all 0's.
        matrix = K.variable(numpy.array([[2, 1, 5, 7], [0, 0, 0, 0]]))
        matrix_mask = K.variable(numpy.array([[0, 0, 0, 0], [0, 0, 0, 0]]))
        matrix_normalized = K.eval(l1_normalize(matrix, matrix_mask))
        assert_array_almost_equal(matrix_normalized,
                                  numpy.array([[0.25, 0.25,
                                                0.25, 0.25],
                                               [0.25, 0.25,
                                                0.25, 0.25]]))
        assert_almost_equal(numpy.array([1.0, 1.0]),
                            numpy.sum(matrix_normalized, axis=1), decimal=6)

    def test_l1_normalize_special_cases(self):
        # Testing the special masked 1D case where the mask
        # all zero and the input is all zero as well.
        vector_1d_zeros = K.variable(numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_zeros_mask = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_normalized = K.eval(l1_normalize(vector_1d_zeros,
                                                         vector_1d_zeros_mask))
        assert_array_almost_equal(vector_1d_zeros_normalized,
                                  numpy.array([[0.25, 0.25, 0.25, 0.25]]))

        # Testing the special masked 1D case where the mask
        # all zero and the input is not all zero.
        vector_1d_zeros = K.variable(numpy.array([[2, 1, 5, 7]]))
        vector_1d_zeros_mask = K.variable(numpy.array([[0, 0, 0, 0]]))
        vector_1d_zeros_normalized = K.eval(l1_normalize(vector_1d_zeros,
                                                         vector_1d_zeros_mask))
        assert_array_almost_equal(vector_1d_zeros_normalized,
                                  numpy.array([[0.25, 0.25, 0.25, 0.25]]))

    def test_masked_softmax_no_mask(self):
        # Testing the general unmasked 1D case.
        vector_1d = K.variable(numpy.array([[1.0, 2.0, 3.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, None))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.090031, 0.244728, 0.665241]]))
        assert_almost_equal(1.0, numpy.sum(vector_1d_softmaxed), decimal=6)

        vector_1d = K.variable(numpy.array([[1.0, 2.0, 5.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, None))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.017148, 0.046613, 0.93624]]))

        # Testing the unmasked 1D case where the input is all 0s.
        vector_zero = K.variable(numpy.array([[0.0, 0.0, 0.0]]))
        vector_zero_softmaxed = K.eval(masked_softmax(vector_zero, None))
        assert_array_almost_equal(vector_zero_softmaxed,
                                  numpy.array([[0.33333334, 0.33333334, 0.33333334]]))

        # Testing the general unmasked batched case.
        matrix = K.variable(numpy.array([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, None))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.09003057, 0.24472847, 0.66524096]]))

        # Testing the unmasked batched case where one of the inputs are all 0s.
        matrix = K.variable(numpy.array([[1.0, 2.0, 5.0], [0.0, 0.0, 0.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, None))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01714783, 0.04661262, 0.93623955],
                                               [0.33333334, 0.33333334, 0.33333334]]))

    def test_masked_softmax_masked(self):
        # Testing the general masked 1D case.
        vector_1d = K.variable(numpy.array([[1.0, 2.0, 5.0]]))
        mask_1d = K.variable(numpy.array([[1.0, 0.0, 1.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, mask_1d))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382]]))

        vector_1d = K.variable(numpy.array([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = K.variable(numpy.array([[1.0, 0.0, 1.0, 1.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, mask_1d))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.01321289, 0.0,
                                                0.26538793, 0.72139918]]))

        # Testing the masked 1D case where the input is all 0s and the mask
        # is not all 0s.
        vector_1d = K.variable(numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = K.variable(numpy.array([[0.0, 0.0, 0.0, 1.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, mask_1d))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0, 0, 0, 1]]))

        # Testing the masked 1D case where the input is not all 0s
        # and the mask is all 0s.
        vector_1d = K.variable(numpy.array([[0.0, 2.0, 3.0, 4.0]]))
        mask_1d = K.variable(numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, mask_1d))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0,
                                                0.0, 0.0]]))

        # Testing the masked 1D case where the input is all 0s and
        # the mask is all 0s.
        vector_1d = K.variable(numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        mask_1d = K.variable(numpy.array([[0.0, 0.0, 0.0, 0.0]]))
        vector_1d_softmaxed = K.eval(masked_softmax(vector_1d, mask_1d))
        assert_array_almost_equal(vector_1d_softmaxed,
                                  numpy.array([[0.0, 0.0,
                                                0.0, 0.0]]))

        # Testing the general masked batched case.
        matrix = K.variable(numpy.array([[1.0, 2.0, 5.0], [1.0, 2.0, 3.0]]))
        mask = K.variable(numpy.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, mask))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.01798621, 0.0, 0.98201382],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s but
        # none of the masks are all 0.
        matrix = K.variable(numpy.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = K.variable(numpy.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, mask))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.090031, 0.244728, 0.665241]]))

        # Testing the masked batch case where one of the inputs is all 0s and
        # one of the masks are all 0.
        matrix = K.variable(numpy.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = K.variable(numpy.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, mask))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.5, 0.0, 0.5],
                                               [0.0, 0.0, 0.0]]))

        matrix = K.variable(numpy.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]]))
        mask = K.variable(numpy.array([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]))
        masked_matrix_softmaxed = K.eval(masked_softmax(matrix, mask))
        assert_array_almost_equal(masked_matrix_softmaxed,
                                  numpy.array([[0.0, 0.0, 0.0],
                                               [0.11920292, 0.0, 0.88079708]]))
