# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K
from keras.layers import Input, Masking
from keras.models import Model

from deep_qa.layers.backend.batch_dot import BatchDot
from deep_qa.layers.wrappers.output_mask import OutputMask
from ...common.test_case import DeepQaTestCase


class TestBatchDotLayer(DeepQaTestCase):
    def test_compute_mask_basic(self):
        batch_size = 2
        # test the case where the tensors are even
        # tensor_a has shape (2, 3, 2), so mask_a has shape (2, 3)
        tensor_a = K.variable(numpy.random.randint(7, size=(batch_size, 3, 2)))
        mask_a = K.variable(numpy.array([[1, 0, 1], [1, 1, 0]]))
        # tensor_b has shape (2, 4, 2), so mask_b has shape (2, 4)
        tensor_b = K.variable(numpy.random.randint(7, size=(batch_size, 4, 2)))
        mask_b = K.variable(numpy.array([[0, 1, 1, 1], [1, 0, 1, 1]]))
        # a_dot_b would have shape (2, 3, 4), so mask of a_dot_b has shape (2, 3, 4)
        calculated_mask = K.eval(BatchDot().compute_mask([tensor_a, tensor_b],
                                                         [mask_a, mask_b]))
        assert_almost_equal(calculated_mask, numpy.array([[[0.0, 1.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0],
                                                           [0.0, 1.0, 1.0, 1.0]],
                                                          [[1.0, 0.0, 1.0, 1.0],
                                                           [1.0, 0.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0]]]))

        # test the case where tensor_a has less dimensions than tensor_b
        # tensor_a has shape (2, 4, 2), so mask_a has shape (2, 4)
        tensor_a = K.variable(numpy.random.randint(7, size=(batch_size, 4, 2)))
        mask_a = K.variable(numpy.array([[1, 0, 1, 0], [1, 1, 0, 0]]))
        # tensor_b has shape (2, 4, 3, 2), so mask_b has shape (2, 4, 3)
        tensor_b = K.variable(numpy.random.randint(7, size=(batch_size, 4, 3, 2)))
        mask_b = K.variable(numpy.array([[[1, 1, 1],
                                          [1, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 0]],
                                         [[1, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 0],
                                          [0, 0, 0]]]))
        # a_dot_b would have shape (2, 4, 3), so mask of a_dot_b has shape (2, 4, 3)
        calculated_mask = K.eval(BatchDot().compute_mask([tensor_a, tensor_b],
                                                         [mask_a, mask_b]))
        assert calculated_mask.shape == (batch_size, 4, 3)
        assert_almost_equal(calculated_mask, numpy.array([[[1.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0],
                                                           [1.0, 1.0, 0.0],
                                                           [0.0, 0.0, 0.0]],
                                                          [[1.0, 1.0, 1.0],
                                                           [1.0, 1.0, 0.0],
                                                           [0.0, 0.0, 0.0],
                                                           [0.0, 0.0, 0.0]]]))

        # test the case where tensor_a has more dimensions than tensor_b
        # tensor_a has shape (2, 3, 4, 2), so mask_a has shape (2, 3, 4)
        tensor_a = K.variable(numpy.random.randint(7, size=(batch_size, 3, 4, 2)))
        mask_a = K.variable(numpy.array([[[1, 1, 1, 1],
                                          [1, 1, 1, 1],
                                          [1, 1, 0, 1]],
                                         [[1, 1, 1, 1],
                                          [1, 1, 0, 1],
                                          [1, 0, 0, 1]]]))
        # tensor_b has shape (2, 3, 2), so mask_b has shape (2, 3)
        tensor_b = K.variable(numpy.random.randint(7, size=(batch_size, 3, 2)))
        mask_b = K.variable(numpy.array([[1, 0, 1], [1, 1, 0]]))
        # a_dot_b would have shape (2, 3, 4), so mask of a_dot_b has shape (2, 3, 4)
        calculated_mask = K.eval(BatchDot().compute_mask([tensor_a, tensor_b],
                                                         [mask_a, mask_b]))
        assert calculated_mask.shape == (batch_size, 3, 4)
        assert_almost_equal(calculated_mask, numpy.array([[[1.0, 1.0, 1.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0],
                                                           [1.0, 1.0, 0.0, 1.0]],
                                                          [[1.0, 1.0, 1.0, 1.0],
                                                           [1.0, 1.0, 0.0, 1.0],
                                                           [0.0, 0.0, 0.0, 0.0]]]))

    def test_a_smaller_than_b(self):
        batch_size = 3
        tensor_a = numpy.random.randint(7, size=(batch_size, 5))
        tensor_b = numpy.random.randint(7, size=(batch_size, 2, 5))

        # Manually set some values to 1 here, which will be masked later
        # (1 and not 0 so that masked values are still non-zero in the output)
        tensor_a[0] = 0
        tensor_b[0][1] = 0

        input_tensor_a = Input(shape=(5,))
        masked_tensor_a = Masking(mask_value=0)(input_tensor_a)
        input_tensor_b = Input(shape=(2, 5))
        masked_tensor_b = Masking(mask_value=0)(input_tensor_b)

        a_dot_b = BatchDot()([masked_tensor_a, masked_tensor_b])

        a_dot_b_mask = OutputMask()(a_dot_b)
        model = Model(input=[input_tensor_a, input_tensor_b],
                      output=[a_dot_b, a_dot_b_mask])
        # a_dot_b and mask_tensor are of shape (3, 2).
        a_dot_b_tensor, mask_tensor = model.predict([tensor_a, tensor_b])
        # Test that the dot happened like we expected.
        for i in range(batch_size):
            # each dot product should be of shape (2,)
            assert_almost_equal(a_dot_b_tensor[i],
                                numpy.einsum("i,mi->m", tensor_a[i], tensor_b[i]))
        # Check that the values in the output mask are 0 where the
        # values were set to 1 above.
        assert mask_tensor[0][0] == 0
        assert mask_tensor[0][1] == 0

    def test_a_larger_than_b(self):
        batch_size = 3
        tensor_a = numpy.random.randint(7, size=(batch_size, 2, 5))
        tensor_b = numpy.random.randint(7, size=(batch_size, 5))

        # Manually set some values to 1 here, which will be masked later
        # (1 and not 0 so that masked values are still non-zero in the output)
        tensor_a[0][1] = 0
        tensor_b[0] = 0

        input_tensor_a = Input(shape=(2, 5))
        masked_tensor_a = Masking(mask_value=0)(input_tensor_a)
        input_tensor_b = Input(shape=(5,))
        masked_tensor_b = Masking(mask_value=0)(input_tensor_b)

        a_dot_b = BatchDot()([masked_tensor_a, masked_tensor_b])

        a_dot_b_mask = OutputMask()(a_dot_b)
        model = Model(input=[input_tensor_a, input_tensor_b],
                      output=[a_dot_b, a_dot_b_mask])
        # a_dot_b and mask_tensor are of shape (3, 2).
        a_dot_b_tensor, mask_tensor = model.predict([tensor_a, tensor_b])
        # Test that the dot happened like we expected.
        for i in range(batch_size):
            # each dot product should be of shape (2,)
            assert_almost_equal(a_dot_b_tensor[i],
                                numpy.einsum("mi,i->m", tensor_a[i], tensor_b[i]))
        # Check that the values in the output mask are 0 where the
        # values were set to 1 above.
        assert mask_tensor[0][0] == 0
        assert mask_tensor[0][1] == 0

    def test_a_smaller_than_b_higher_dimension(self):
        batch_size = 3
        tensor_a = numpy.random.randint(7, size=(batch_size, 4, 5))
        tensor_b = numpy.random.randint(7, size=(batch_size, 4, 2, 5))

        # Manually set some values to 1 here, which will be masked later
        # (1 and not 0 so that masked values are still non-zero in the output)
        tensor_a[0][1] = 0
        tensor_a[1][3] = 0
        tensor_b[0][1][1] = 0
        tensor_b[0][2][1] = 0

        input_tensor_a = Input(shape=(4, 5))
        masked_tensor_a = Masking(mask_value=0)(input_tensor_a)
        input_tensor_b = Input(shape=(4, 2, 5))
        masked_tensor_b = Masking(mask_value=0)(input_tensor_b)

        if K.backend() == "theano":
            self.assertRaises(RuntimeError, BatchDot(),
                              [masked_tensor_a, masked_tensor_b])
            return
        else:
            a_dot_b = BatchDot()([masked_tensor_a, masked_tensor_b])

        a_dot_b_mask = OutputMask()(a_dot_b)
        model = Model(input=[input_tensor_a, input_tensor_b],
                      output=[a_dot_b, a_dot_b_mask])
        # a_dot_b and mask_tensor are of shape (3, 4, 2).
        a_dot_b_tensor, mask_tensor = model.predict([tensor_a, tensor_b])
        # Test that the dot happened like we expected.
        for i in range(batch_size):
            # each dot product should be of shape (4, 2)
            assert_almost_equal(a_dot_b_tensor[i],
                                numpy.einsum("ij,imj->im", tensor_a[i], tensor_b[i]))
        # Check that the values in the output mask are 0 where the
        # values were set to 1 above.
        assert mask_tensor[0][1][0] == 0
        assert mask_tensor[0][1][1] == 0
        assert mask_tensor[0][2][1] == 0
        assert mask_tensor[1][3][0] == 0
        assert mask_tensor[1][3][1] == 0

    def test_a_larger_than_b_higher_dimension(self):
        batch_size = 3
        tensor_a = numpy.random.randint(7, size=(batch_size, 4, 2, 5))
        tensor_b = numpy.random.randint(7, size=(batch_size, 4, 5))

        # Manually set some values to 1 here, which will be masked later
        # (1 and not 0 so that masked values are still non-zero in the output)
        tensor_a[0][1][1] = 0
        tensor_a[0][2][1] = 0
        tensor_b[0][1] = 0
        tensor_b[1][3] = 0

        input_tensor_a = Input(shape=(4, 2, 5))
        masked_tensor_a = Masking(mask_value=0)(input_tensor_a)
        input_tensor_b = Input(shape=(4, 5))
        masked_tensor_b = Masking(mask_value=0)(input_tensor_b)

        if K.backend() == "theano":
            self.assertRaises(RuntimeError, BatchDot(),
                              [masked_tensor_a, masked_tensor_b])
            return
        else:
            a_dot_b = BatchDot()([masked_tensor_a, masked_tensor_b])
        a_dot_b_mask = OutputMask()(a_dot_b)
        model = Model(input=[input_tensor_a, input_tensor_b],
                      output=[a_dot_b, a_dot_b_mask])
        # a_dot_b and mask_tensor are of shape (3, 4, 2).
        a_dot_b_tensor, mask_tensor = model.predict([tensor_a, tensor_b])
        # Test that the dot happened like we expected.
        for i in range(batch_size):
            # each dot product should be of shape (4, 2)
            assert_almost_equal(a_dot_b_tensor[i],
                                numpy.einsum("imj,ij->im", tensor_a[i], tensor_b[i]))
            # Check that the values in the output mask are 0 where the
            # values were set to 1 above.
        assert mask_tensor[0][1][0] == 0
        assert mask_tensor[0][1][1] == 0
        assert mask_tensor[0][2][1] == 0
        assert mask_tensor[1][3][0] == 0
        assert mask_tensor[1][3][1] == 0

    def test_output_shapes(self):
        bd = BatchDot()
        a_shapes = [(5, 10), (1, 1, 1), (1, 5, 3), (1, 5, 4, 3), (1, 5, 3)]
        b_shapes = [(5, 10), (1, 1, 1), (1, 2, 3), (1, 5, 3), (1, 5, 4, 3)]
        expected_shapes = [(5, 1), (1, 1, 1), (1, 5, 2), (1, 5, 4), (1, 5, 4)]
        for a_shape, b_shape, expected_shape in zip(a_shapes, b_shapes, expected_shapes):
            if (len(a_shape) > 3 or len(b_shape) > 3) and K.backend() == "theano":
                # this breaks in theano, so check that an error is raised
                self.assertRaises(RuntimeError, bd.call,
                                  [K.ones(shape=a_shape), K.ones(shape=b_shape)])
            else:
                assert K.eval(bd([K.ones(shape=a_shape), K.ones(shape=b_shape)])).shape == expected_shape
            assert bd.get_output_shape_for([a_shape, b_shape]) == expected_shape
