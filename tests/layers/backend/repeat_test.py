# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend import Repeat

class TestRepeatLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        input_length = 3
        repetitions = 4
        input_layer = Input(shape=(input_length,), dtype='float32')
        repeat_output = Repeat(axis=1, repetitions=repetitions)(input_layer)
        model = Model(inputs=[input_layer], outputs=[repeat_output])
        input_tensor = numpy.asarray([[2, 5, 3], [-1, -4, -2]])
        repeat_tensor = model.predict([input_tensor])
        assert repeat_tensor.shape == (batch_size, repetitions, input_length)
        for i in range(repetitions):
            numpy.testing.assert_almost_equal(repeat_tensor[:, i, :], [[2, 5, 3], [-1, -4, -2]])
