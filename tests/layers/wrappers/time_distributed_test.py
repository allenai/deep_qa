# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_array_almost_equal
from keras.layers import Input, Lambda
from keras.models import Model

from deep_qa.layers.wrappers import TimeDistributed
from ...common.test_case import DeepQaTestCase

class TestTimeDistributed(DeepQaTestCase):
    def test_handles_multiple_inputs(self):
        input_layer_1 = Input(shape=(3, 1), dtype='int32')
        input_layer_2 = Input(shape=(3, 1), dtype='int32')
        combine_layer = Lambda(lambda x: x[0] ** x[1] + 1,
                               output_shape=lambda x: (x[0][0], 1),
                               name="a^b + 1 Layer")
        td_combine = TimeDistributed(combine_layer)
        output = td_combine([input_layer_1, input_layer_2])
        model = Model([input_layer_1, input_layer_2], output)

        batch_input_1 = numpy.array([[[4], [5], [6]],
                                     [[3], [3], [3]],
                                     [[0], [1], [2]]], dtype='float32')
        batch_input_2 = numpy.array([[[3], [2], [1]],
                                     [[1], [2], [3]],
                                     [[1], [0], [2]]], dtype='float32')

        expected_result = (batch_input_1 ** batch_input_2 + 1)
        # In TimeDistributed, we reshape tensors whose final dimension is 1, so we need to do that here.
        if numpy.shape(expected_result)[-1] == 1:
            expected_result = numpy.reshape(expected_result, numpy.shape(expected_result)[:-1])
        result = model.predict([batch_input_1, batch_input_2])
        assert_array_almost_equal(result, expected_result)
