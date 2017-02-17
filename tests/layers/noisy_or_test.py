# pylint: disable=no-self-use, invalid-name

from unittest import TestCase
import numpy as np
from numpy.testing import assert_array_almost_equal

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from deep_qa.layers.noisy_or import NoisyOr, BetweenZeroAndOne

class TestNoisyOr(TestCase):
    def test_general_case(self):

        input_layer = Input(shape=(3, 2,), dtype='float32', name="input")
        axis = 2
        noisy_or_layer = NoisyOr(axis=axis)
        output = noisy_or_layer(input_layer)
        model = Model([input_layer], output)

        # Testing general unmasked batched case.
        q = K.eval(noisy_or_layer.noise_parameter)
        batch_original_data = np.array([[[0.2, 0.1],
                                         [0.5, 0.3],
                                         [0.3, 0.7]],
                                        [[0.4, 0.55],
                                         [0.65, 0.8],
                                         [0.9, 0.15]]])
        batch_result = model.predict([batch_original_data])
        batch_desired_result = 1.0 - np.prod(1.0 - (q * batch_original_data), axis=axis)
        assert_array_almost_equal(batch_result, batch_desired_result)


    def test_between_zero_and_one_constraint(self):
        p = K.variable(np.asarray([0.35, -0.4, 1.0, 1.2]), dtype='float32')
        desired_result = np.asarray([0.35, K.epsilon(), 1.0, 1.0])
        result = K.eval(BetweenZeroAndOne()(p))
        assert_array_almost_equal(result, desired_result)
