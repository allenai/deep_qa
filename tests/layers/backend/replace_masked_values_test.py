# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend import AddMask, ReplaceMaskedValues

class TestReplaceMaskedValues:
    def test_call_works_on_simple_input(self):
        input_length = 3
        input_layer = Input(shape=(input_length,), dtype='float32')
        masked = AddMask(2)(input_layer)
        replaced = ReplaceMaskedValues(4)(masked)
        model = Model(inputs=[input_layer], outputs=[replaced])
        input_tensor = numpy.asarray([[2, 5, 2], [2, -4, -2]])
        replaced_tensor = model.predict([input_tensor])
        assert_almost_equal(replaced_tensor, numpy.asarray([[4, 5, 4], [4, -4, -2]]))
