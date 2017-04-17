# pylint: disable=no-self-use
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import keras.backend as K
from keras.layers import Input
from keras.models import Model

from deep_qa.layers import OptionAttentionSum
from deep_qa.common.checks import ConfigurationError
from ..common.test_case import DeepQaTestCase


class TestOptionAttentionSum(DeepQaTestCase):
    def test_mean_mode(self):
        document_probabilities_length = 6
        document_indices_length = document_probabilities_length
        max_num_options = 3
        max_num_words_per_option = 2

        document_indices_input = Input(shape=(document_indices_length,),
                                       dtype='int32',
                                       name="document_indices_input")
        document_probabilities_input = Input(shape=(document_probabilities_length,),
                                             dtype='float32',
                                             name="document_probabilities_input")
        options_input = Input(shape=(max_num_options, max_num_words_per_option),
                              dtype='int32', name="options_input")
        option_attention_sum_mean = OptionAttentionSum()([document_indices_input,
                                                          document_probabilities_input,
                                                          options_input])
        model = Model([document_indices_input,
                       document_probabilities_input,
                       options_input],
                      option_attention_sum_mean)

        document_indices = np.array([[1, 2, 3, 4, 1, 2]])
        document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03]])

        # Testing the general single-batch case.
        options = np.array([[[1, 2], [3, 4], [1, 2]]])
        result = model.predict([document_indices, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.17, 0.35, 0.17]]))

        options = np.array([[[1, 1], [3, 1], [4, 2]]])
        result = model.predict([document_indices, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.11, 0.205, 0.315]]))

        # Testing the general batch case.
        batch_document_indices = np.array([[1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 1, 2]])
        batch_document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03],
                                                 [.1, .2, .3, .4, 0.01, 0.03]])
        batch_options = np.array([[[1, 2], [3, 4], [1, 2]], [[1, 1], [3, 1], [4, 2]]])
        result = model.predict([batch_document_indices, batch_document_probabilities,
                                batch_options])
        assert_array_almost_equal(result, np.array([[0.17, 0.35, 0.17],
                                                    [0.11, 0.205, 0.315]]))

    def test_mean_mode_mask(self):
        # Testing the general masked batched case.
        document_indices = K.variable(np.array([[1, 2, 3, 4, 1, 2]]))
        document_probabilities = K.variable(np.array([[.1, .2, .3, .4, 0.01, 0.03]]))
        options = K.variable(np.array([[[1, 2, 1], [3, 4, 2], [4, 1, 0]]]))
        option_attention_sum_mean = K.eval(OptionAttentionSum().call([document_indices,
                                                                      document_probabilities,
                                                                      options]))
        assert_array_almost_equal(option_attention_sum_mean,
                                  np.array([[0.14999999, 0.31000003, 0.255]]))

        options = K.variable(np.array([[[1, 2, 1], [3, 4, 2], [0, 0, 0]]]))
        option_attention_sum_mean = K.eval(OptionAttentionSum().call([document_indices,
                                                                      document_probabilities,
                                                                      options]))
        assert_array_almost_equal(option_attention_sum_mean,
                                  np.array([[0.14999999, 0.31000003, 0.0]]))

        # Testing the masked batched case where input is all 0s.
        options = K.variable(np.array([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]))
        option_attention_sum_mean = K.eval(OptionAttentionSum().call([document_indices,
                                                                      document_probabilities,
                                                                      options]))
        assert_array_almost_equal(option_attention_sum_mean,
                                  np.array([[0, 0, 0]]))

    def test_sum_mode(self):
        document_probabilities_length = 6
        document_indices_length = document_probabilities_length
        max_num_options = 3
        max_num_words_per_option = 2

        document_indices_input = Input(shape=(document_indices_length,),
                                       dtype='int32',
                                       name="document_indices_input")
        document_probabilities_input = Input(shape=(document_probabilities_length,),
                                             dtype='float32',
                                             name="document_probabilities_input")
        options_input = Input(shape=(max_num_options, max_num_words_per_option),
                              dtype='int32', name="options_input")
        option_attention_sum_mean = OptionAttentionSum("sum")([document_indices_input,
                                                               document_probabilities_input,
                                                               options_input])
        model = Model([document_indices_input,
                       document_probabilities_input,
                       options_input],
                      option_attention_sum_mean)

        document_indices = np.array([[1, 2, 3, 4, 1, 2]])
        document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03]])

        # Testing the general single-batch case.
        options = np.array([[[1, 2], [3, 4], [1, 2]]])
        result = model.predict([document_indices, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.34, 0.70, 0.34]]))

        options = np.array([[[1, 1], [3, 1], [4, 2]]])
        result = model.predict([document_indices, document_probabilities, options])
        assert_array_almost_equal(result, np.array([[0.22, 0.41, 0.63]]))

        # Testing the general batch case
        batch_document_indices = np.array([[1, 2, 3, 4, 1, 2], [1, 2, 3, 4, 1, 2]])
        batch_document_probabilities = np.array([[.1, .2, .3, .4, 0.01, 0.03],
                                                 [.1, .2, .3, .4, 0.01, 0.03]])
        batch_options = np.array([[[1, 2], [3, 4], [1, 2]], [[1, 1], [3, 1], [4, 2]]])
        result = model.predict([batch_document_indices, batch_document_probabilities,
                                batch_options])
        assert_array_almost_equal(result, np.array([[0.34, 0.70, 0.34],
                                                    [0.22, 0.41, 0.63]]))

    def test_multiword_option_mode_validation(self):
        self.assertRaises(ConfigurationError, OptionAttentionSum, "summean")

    def test_compute_mask(self):
        option_attention_sum = OptionAttentionSum()
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [2, 3, 3],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 1, 0, 0]]))
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [1, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 1, 0, 0]]))
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [0, 0, 0],
                                                                          [0, 0, 0], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 0, 0, 0]]))

        # test batch case
        result = option_attention_sum.compute_mask(["_", "_",
                                                    K.variable(np.array([[[1, 2, 0], [2, 3, 3],
                                                                          [0, 0, 0], [0, 0, 0]],
                                                                         [[1, 1, 0], [3, 3, 3],
                                                                          [0, 0, 3], [0, 0, 0]]],
                                                                        dtype="int32"))])
        assert_array_equal(K.eval(result), np.array([[1, 1, 0, 0], [1, 1, 1, 0]]))
