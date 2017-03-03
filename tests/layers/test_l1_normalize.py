# pylint: disable=no-self-use
import numpy as np
from numpy.testing import assert_array_almost_equal

from keras.layers import Input, Masking
from keras.models import Model
from deep_qa.layers.l1_normalize import L1Normalize
from ..common.test_case import DeepQaTestCase


class TestL1Normalize(DeepQaTestCase):
    def test_general_case(self):
        input_length = 6

        input_layer = Input(shape=(input_length,), dtype='float32', name="input")
        l1_normalize_layer = L1Normalize()
        normalized_input = l1_normalize_layer(input_layer)

        model = Model([input_layer], normalized_input)
        # Testing general unmasked 1D case.
        unnormalized_vector = np.array([[.1, .2, .3, .4, 0.01, 0.03]])
        result = model.predict([unnormalized_vector])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(1))

        # Testing general unmasked batched case.
        unnormalized_matrix = np.array([[.1, .2, .3, .4, 0.01, 0.03],
                                        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
        result = model.predict([unnormalized_matrix])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615],
                                                    [1.0/21.0, 2.0/21.0, 3.0/21.0,
                                                     4.0/21.0, 5.0/21.0, 6.0/21.0]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(2))

    def test_squeeze_case_mask(self):
        input_length = 4
        mask_value = 3

        input_layer = Input(shape=(input_length, 1), dtype='float32', name="input")
        mask_layer = Masking(mask_value=mask_value)
        masked_input = mask_layer(input_layer)
        l1_normalize_layer = L1Normalize()
        normalized_input = l1_normalize_layer(masked_input)

        model = Model([input_layer], normalized_input)
        unnormalized_vector = np.array([[[1.0], [2.0], [3.0], [4.0]]])
        result = model.predict([unnormalized_vector])
        assert_array_almost_equal(result, np.array([[0.14285715, 0.2857143,
                                                     0, 0.5714286]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(1))

        # Testing general masked batched case
        unnormalized_matrix = np.array([[[1.0], [2.0], [3.0], [4.0]],
                                        [[3.0], [2.0], [3.0], [4.0]]])

        result = model.predict([unnormalized_matrix])
        assert_array_almost_equal(result, np.array([[0.14285715, 0.2857143,
                                                     0, 0.5714286],
                                                    [0, 2.0/6.0, 0, 4.0/6.0]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(2))

    def test_squeeze_case(self):
        input_length = 6

        input_layer = Input(shape=(input_length, 1), dtype='float32', name="input")
        l1_normalize_layer = L1Normalize()
        normalized_input = l1_normalize_layer(input_layer)

        model = Model([input_layer], normalized_input)
        unnormalized_vector = np.array([[[.1], [.2], [.3], [.4], [0.01], [0.03]]])
        result = model.predict([unnormalized_vector])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(1))

        # Testing general unmasked batched case.
        unnormalized_matrix = np.array([[[.1], [.2], [.3], [.4], [0.01], [0.03]],
                                        [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]]])
        result = model.predict([unnormalized_matrix])
        assert_array_almost_equal(result, np.array([[0.09615385, 0.1923077,
                                                     0.28846157, 0.38461539,
                                                     0.00961538, 0.02884615],
                                                    [1.0/21.0, 2.0/21.0, 3.0/21.0,
                                                     4.0/21.0, 5.0/21.0, 6.0/21.0]]))
        assert_array_almost_equal(np.sum(result, axis=1), np.ones(2))
