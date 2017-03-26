# pylint: disable=no-self-use,invalid-name

import numpy
from keras import backend as K
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.attention.masked_softmax import MaskedSoftmax

class TestMaskedSoftmaxLayer:
    def test_call_works_with_no_mask(self):
        batch_size = 1
        num_options = 4
        options_input = Input(shape=(num_options,), dtype='float32')
        softmax_result = MaskedSoftmax()(options_input)
        model = Model(inputs=[options_input], outputs=[softmax_result])
        options_tensor = numpy.asarray([[2, 4, 0, 1]])
        softmax_tensor = model.predict([options_tensor])
        assert softmax_tensor.shape == (batch_size, num_options)
        numpy.testing.assert_almost_equal(softmax_tensor,
                                          [[0.112457, 0.830953, 0.015219, 0.041371]],
                                          decimal=5)

    def test_call_handles_higher_order_input(self):
        batch_size = 1
        length_1 = 5
        length_2 = 3
        num_options = 4
        options_input = Input(shape=(length_1, length_2, num_options,), dtype='float32')
        softmax_result = MaskedSoftmax()(options_input)
        model = Model(inputs=[options_input], outputs=[softmax_result])
        options_tensor = numpy.zeros((batch_size, length_1, length_2, num_options))
        for i in range(length_1):
            for j in range(length_2):
                options_tensor[0, i, j] = [2, 4, 0, 1]
        softmax_tensor = model.predict([options_tensor])
        assert softmax_tensor.shape == (batch_size, length_1, length_2, num_options)
        for i in range(length_1):
            for j in range(length_2):
                numpy.testing.assert_almost_equal(softmax_tensor[0, i, j],
                                                  [0.112457, 0.830953, 0.015219, 0.041371],
                                                  decimal=5)

    def test_call_handles_masking_properly(self):
        options = K.variable(numpy.asarray([[2, 4, 0, 1]]))
        mask = K.variable(numpy.asarray([[1, 0, 1, 1]]))
        softmax = K.eval(MaskedSoftmax().call(options, mask=mask))
        assert softmax.shape == (1, 4)
        numpy.testing.assert_almost_equal(softmax, [[0.66524096, 0, 0.09003057, 0.24472847]])
