# pylint: disable=no-self-use, invalid-name
import numpy as np
from numpy.testing import assert_array_almost_equal

from keras import backend as K
from keras.layers import Input
from keras.models import Model
from deep_qa.layers import NoisyOr, BetweenZeroAndOne
from ..common.test_case import DeepQaTestCase

class TestNoisyOr(DeepQaTestCase):
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

        # Testing the masked case.
        # Here's a modified version of the batch_original_data, with extra probabilities.
        batch_data_with_masks = K.variable(np.array([[[0.2, 0.1, 0.7], [0.5, 0.3, 0.3], [0.3, 0.7, 0.2]],
                                                     [[0.4, 0.55, 0.3], [0.65, 0.8, 0.1], [0.9, 0.15, 0.0]]]),
                                           dtype="float32")
        # Now here the added 3rd element is masked out, so the noisy_or probabilities resulting from the
        # masked version should be the same as the unmasked one (above).
        masks = K.variable(np.array([[[1, 1, 0], [1, 1, 0], [1, 1, 0]],
                                     [[1, 1, 0], [1, 1, 0], [1, 1, 0]]]), dtype="float32")

        masking_results = K.eval(noisy_or_layer.call(inputs=batch_data_with_masks, mask=masks))
        assert_array_almost_equal(batch_result, masking_results)

    def test_between_zero_and_one_constraint(self):
        p = K.variable(np.asarray([0.35, -0.4, 1.0, 1.2]), dtype='float32')
        desired_result = np.asarray([0.35, K.epsilon(), 1.0, 1.0])
        result = K.eval(BetweenZeroAndOne()(p))
        assert_array_almost_equal(result, desired_result)
