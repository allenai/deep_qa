# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend import RepeatLike

class TestRepeatLikeLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        input_length = 3
        repetitions = 4
        input_layer = Input(shape=(input_length,), dtype='float32')
        input_layer_2 = Input(shape=(None,), dtype='float32')
        repeat_output = RepeatLike(axis=1, copy_from_axis=1)([input_layer, input_layer_2])
        model = Model(inputs=[input_layer, input_layer_2], outputs=[repeat_output])
        input_tensor = numpy.asarray([[2, 5, 3], [-1, -4, -2]])
        input_tensor_2 = numpy.ones((batch_size, repetitions))
        repeat_tensor = model.predict([input_tensor, input_tensor_2])
        assert repeat_tensor.shape == (batch_size, repetitions, input_length)
        for i in range(repetitions):
            numpy.testing.assert_almost_equal(repeat_tensor[:, i, :], [[2, 5, 3], [-1, -4, -2]])
