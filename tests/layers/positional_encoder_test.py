# pylint: disable=no-self-use,invalid-name
from keras.layers import Input, Embedding
from keras.models import Model
import numpy as np
from deep_qa.layers.encoders import PositionalEncoder


class TestPositionalEncoder:
    def test_on_unmasked_input(self):
        sentence_length = 3
        embedding_dim = 3
        vocabulary_size = 5
        # Manual embedding vectors so we can compute exact values for test.
        embedding_weights = np.asarray([[0.0, 0.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0]])
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, weights=[embedding_weights],
                              output_dim=embedding_dim, mask_zero=True)
        encoder = PositionalEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")

        test_input = np.asarray([[1, 2, 3]])
        actual_output = model.predict(test_input)[0]
        expected_output = np.asarray([1.3333333, 1.6666666, 2])
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input(self):

        sentence_length = 5
        embedding_dim = 3
        vocabulary_size = 5
        # Manual embedding vectors so we can compute exact values for test.
        embedding_weights = np.asarray([[0.0, 0.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0]])
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, weights=[embedding_weights],
                              output_dim=embedding_dim, mask_zero=True)
        encoder = PositionalEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")

        test_input = np.asarray([[0, 1, 2, 3, 0]])
        actual_output = model.predict(test_input)[0]
        expected_output = np.asarray([1.3333333, 1.6666666, 2])
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_completely_masked_input(self):

        sentence_length = 5
        embedding_dim = 3
        vocabulary_size = 5
        embedding_weights = np.asarray([[0.0, 0.0, 0.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [1.0, 1.0, 1.0],
                                        [2.0, 2.0, 2.0]])
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, weights=[embedding_weights],
                              output_dim=embedding_dim, mask_zero=True)
        encoder = PositionalEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(inputs=input_layer, outputs=encoded_input)
        model.compile(loss="mse", optimizer="sgd")

        test_input = np.asarray([[0, 0, 0, 0, 0]])
        actual_output = model.predict(test_input)[0]
        expected_output = np.asarray([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(expected_output, actual_output)
