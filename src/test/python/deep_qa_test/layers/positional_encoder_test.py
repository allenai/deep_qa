# pylint: disable=no-self-use,invalid-name
from keras.layers import Input, Embedding
from keras.models import Model
import numpy as np
from deep_qa.layers.encoders import PositionalEncoder


class TestPositionalEncoder:
    def test_on_unmasked_input(self):
        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size)
        encoder = PositionalEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(input=input_layer, output=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = np.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        expected_output = self.numpy_positional_encoder(embedding_weights[test_input])
        actual_output = model.predict(test_input)
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input(self):
        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, mask_zero=True)
        encoder = PositionalEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(input=input_layer, output=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = np.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        # Omitting the first element (0), because that is supposed to be masked in the model.
        mask = np.ones_like(test_input)
        mask[0, 0] = np.zeros_like(mask[0, 0])
        expected_output = self.numpy_positional_encoder(embedding_weights[test_input], mask)
        actual_output = model.predict(test_input)
        np.testing.assert_array_almost_equal(expected_output, actual_output)

    def numpy_positional_encoder(self, x, mask=None):
        if mask is None:
            ones_like_x = np.ones_like(x)
        else:
            float_mask = mask.astype(float)
            ones_like_x = np.ones_like(x) * np.expand_dims(float_mask, 2)

        # This is an odd way to get the number of words(ie the first dimension of x).
        # However, if the input is masked, using the dimension directly does not
        # equate to the correct number of words. We fix this by adding up a relevant
        # row of ones which has been masked if required.
        masked_m = np.expand_dims(np.sum(ones_like_x, 1), 1)
        one_over_m = ones_like_x / masked_m
        j_index = np.cumsum(ones_like_x, 1)
        d_over_D = np.cumsum(ones_like_x, 2) / np.shape(x)[2]
        one_minus_j = ones_like_x - j_index
        one_minus_two_j = ones_like_x - 2 * j_index

        l_weighting_vectors = (one_minus_j * one_over_m) - \
                              (d_over_D * (one_minus_two_j * one_over_m))

        return np.sum(l_weighting_vectors * x, 1)
