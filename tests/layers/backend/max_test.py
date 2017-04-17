# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend import Max

class TestMaxLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        input_length = 5
        input_layer = Input(shape=(input_length,), dtype='float32')
        max_output = Max()(input_layer)
        model = Model(inputs=[input_layer], outputs=[max_output])
        input_tensor = numpy.asarray([[2, 5, 3, 1, -4], [-1, -4, -2, -10, -4]])
        max_tensor = model.predict([input_tensor])
        assert max_tensor.shape == (batch_size,)
        numpy.testing.assert_almost_equal(max_tensor, [5, -1])
