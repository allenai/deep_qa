# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend import Permute

class TestPermuteLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 2
        input_length_1 = 2
        input_length_2 = 1
        input_layer = Input(shape=(input_length_1, input_length_2), dtype='float32')
        permute_output = Permute(pattern=[0, 2, 1])(input_layer)
        model = Model(inputs=[input_layer], outputs=[permute_output])
        input_tensor = numpy.asarray([[[2], [5]], [[-1], [-4]]])
        permute_tensor = model.predict([input_tensor])
        assert permute_tensor.shape == (batch_size, input_length_2, input_length_1)
        numpy.testing.assert_almost_equal(permute_tensor, [[[2, 5]], [[-1, -4]]])
