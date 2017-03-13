# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend.add_mask import AddMask
from deep_qa.layers.backend.multiply import Multiply
from deep_qa.layers.wrappers.output_mask import OutputMask

class TestMultiply:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        input_length = 5
        input_1_layer = Input(shape=(input_length,), dtype='float32')
        input_2_layer = Input(shape=(input_length,), dtype='float32')
        masking_layer = AddMask()
        masked_input_1 = masking_layer(input_1_layer)
        masked_input_2 = masking_layer(input_2_layer)
        multiply_output = Multiply()([masked_input_1, masked_input_2])
        multiply_mask = OutputMask()(multiply_output)
        model = Model(input=[input_1_layer, input_2_layer], output=[multiply_output, multiply_mask])
        input_1_tensor = numpy.asarray([[2, 5, 0, 1, -4],
                                        [-1, 0, -2, -10, -4]])
        input_2_tensor = numpy.asarray([[3, 2, 1, 0, -2],
                                        [0, 2, 2, 2, 2]])
        multiply_tensor, mask_tensor = model.predict([input_1_tensor, input_2_tensor])
        assert multiply_tensor.shape == (batch_size, input_length)
        numpy.testing.assert_almost_equal(multiply_tensor, [[6, 10, 0, 0, 8],
                                                            [0, 0, -4, -20, -8]])
        numpy.testing.assert_almost_equal(mask_tensor, [[1, 1, 0, 0, 1],
                                                        [0, 0, 1, 1, 1]])

    def test_call_works_with_uneven_dims(self):
        batch_size = 1
        input_length = 2
        input_length_2 = 5
        input_1_layer = Input(shape=(input_length, input_length_2), dtype='float32')
        input_2_layer = Input(shape=(input_length,), dtype='float32')
        masking_layer = AddMask()
        masked_input_1 = masking_layer(input_1_layer)
        masked_input_2 = masking_layer(input_2_layer)
        multiply_output = Multiply()([masked_input_1, masked_input_2])
        multiply_mask = OutputMask()(multiply_output)
        model = Model(input=[input_1_layer, input_2_layer], output=[multiply_output, multiply_mask])
        input_1_tensor = numpy.asarray([[[2, 5, 0, 1, -4],
                                         [-1, 0, -2, -10, -4]]])
        input_2_tensor = numpy.asarray([[2, 1]])
        multiply_tensor, mask_tensor = model.predict([input_1_tensor, input_2_tensor])
        assert multiply_tensor.shape == (batch_size, input_length, input_length_2)
        numpy.testing.assert_almost_equal(multiply_tensor, [[[4, 10, 0, 2, -8],
                                                             [-1, 0, -2, -10, -4]]])
        numpy.testing.assert_almost_equal(mask_tensor, [[[1, 1, 0, 1, 1],
                                                         [1, 0, 1, 1, 1]]])
