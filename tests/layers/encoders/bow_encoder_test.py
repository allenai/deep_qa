# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input, Embedding
from keras.models import Model

from deep_qa.layers.encoders import BOWEncoder

class TestBOWEncoder:
    def test_on_unmasked_input(self):
        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size)
        encoder = BOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(input=input_layer, output=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        expected_output = numpy.mean(embedding_weights[test_input], axis=1)
        actual_output = model.predict(test_input)
        # Exact comparison of the two arrays may break because theano's floating point operations
        # usually have an epsilon. The following comparison is done till the sixth decimal, hence good enough.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_masked_input(self):
        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, mask_zero=True)
        encoder = BOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(input=input_layer, output=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        embedding_weights = embedding.get_weights()[0]  # get_weights returns a list with one element.
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.mean(embedding_weights[test_input[:, 1:]], axis=1)
        actual_output = model.predict(test_input)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)

    def test_on_all_zeros(self):
        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        # Embedding masks zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size, mask_zero=True)
        encoder = BOWEncoder()
        embedded_input = embedding(input_layer)
        encoded_input = encoder(embedded_input)
        model = Model(input=input_layer, output=encoded_input)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 0, 0, 0, 0]], dtype='int32')
        # Omitting the first element (0), because that is supposed to be masked in the model.
        expected_output = numpy.zeros((1, embedding_size))
        actual_output = model.predict(test_input)
        # Following comparison is till the sixth decimal.
        numpy.testing.assert_array_almost_equal(expected_output, actual_output)
