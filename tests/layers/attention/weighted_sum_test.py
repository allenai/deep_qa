# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Embedding, Input
from keras.models import Model

from deep_qa.layers.attention import WeightedSum

class TestWeightedSumLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 1
        sentence_length = 5
        embedding_dim = 4
        matrix_input = Input(shape=(sentence_length, embedding_dim), dtype='float32')
        attention_input = Input(shape=(sentence_length,), dtype='float32')
        aggregated_vector = WeightedSum()([matrix_input, attention_input])
        model = Model(inputs=[matrix_input, attention_input], outputs=[aggregated_vector])
        sentence_tensor = numpy.random.rand(batch_size, sentence_length, embedding_dim)
        attention_tensor = numpy.asarray([[.3, .4, .1, 0, 1.2]])
        aggregated_tensor = model.predict([sentence_tensor, attention_tensor])
        assert aggregated_tensor.shape == (batch_size, embedding_dim)
        expected_tensor = (0.3 * sentence_tensor[0, 0] +
                           0.4 * sentence_tensor[0, 1] +
                           0.1 * sentence_tensor[0, 2] +
                           0.0 * sentence_tensor[0, 3] +
                           1.2 * sentence_tensor[0, 4])
        numpy.testing.assert_almost_equal(aggregated_tensor, [expected_tensor], decimal=5)

    def test_call_handles_higher_order_input(self):
        batch_size = 1
        length_1 = 5
        length_2 = 6
        length_3 = 2
        embedding_dim = 4
        matrix_input = Input(shape=(length_1, length_2, length_3, embedding_dim), dtype='float32')
        attention_input = Input(shape=(length_1, length_2, length_3,), dtype='float32')
        aggregated_vector = WeightedSum()([matrix_input, attention_input])
        model = Model(inputs=[matrix_input, attention_input], outputs=[aggregated_vector])
        sentence_tensor = numpy.random.rand(batch_size, length_1, length_2, length_3, embedding_dim)
        attention_tensor = numpy.random.rand(batch_size, length_1, length_2, length_3)
        aggregated_tensor = model.predict([sentence_tensor, attention_tensor])
        assert aggregated_tensor.shape == (batch_size, length_1, length_2, embedding_dim)
        expected_tensor = (attention_tensor[0, 3, 2, 0] * sentence_tensor[0, 3, 2, 0] +
                           attention_tensor[0, 3, 2, 1] * sentence_tensor[0, 3, 2, 1])
        numpy.testing.assert_almost_equal(aggregated_tensor[0, 3, 2], expected_tensor, decimal=5)

    def test_call_handles_uneven_higher_order_input(self):
        batch_size = 1
        length_1 = 5
        length_2 = 6
        length_3 = 2
        embedding_dim = 4
        matrix_input = Input(shape=(length_3, embedding_dim), dtype='float32')
        attention_input = Input(shape=(length_1, length_2, length_3,), dtype='float32')
        aggregated_vector = WeightedSum()([matrix_input, attention_input])
        model = Model(inputs=[matrix_input, attention_input], outputs=[aggregated_vector])
        sentence_tensor = numpy.random.rand(batch_size, length_3, embedding_dim)
        attention_tensor = numpy.random.rand(batch_size, length_1, length_2, length_3)
        aggregated_tensor = model.predict([sentence_tensor, attention_tensor])
        assert aggregated_tensor.shape == (batch_size, length_1, length_2, embedding_dim)
        for i in range(length_1):
            for j in range(length_2):
                expected_tensor = (attention_tensor[0, i, j, 0] * sentence_tensor[0, 0] +
                                   attention_tensor[0, i, j, 1] * sentence_tensor[0, 1])
                numpy.testing.assert_almost_equal(aggregated_tensor[0, i, j], expected_tensor,
                                                  decimal=5)

    def test_call_handles_masking_properly(self):
        batch_size = 1
        vocab_size = 4
        sentence_length = 5
        embedding_dim = 4
        embedding_weights = numpy.random.rand(vocab_size, embedding_dim)
        embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_weights], mask_zero=True)

        sentence_input = Input(shape=(sentence_length,), dtype='int32')
        sentence_embedding = embedding(sentence_input)
        attention_input = Input(shape=(sentence_length,), dtype='float32')
        aggregated_vector = WeightedSum()([sentence_embedding, attention_input])
        model = Model(inputs=[sentence_input, attention_input], outputs=[aggregated_vector])

        sentence_tensor = numpy.asarray([[1, 3, 2, 1, 0]])
        attention_tensor = numpy.asarray([[.3, .4, .1, 0, 1.2]])
        aggregated_tensor = model.predict([sentence_tensor, attention_tensor])
        assert aggregated_tensor.shape == (batch_size, embedding_dim)
        expected_tensor = (0.3 * embedding_weights[1] +
                           0.4 * embedding_weights[3] +
                           0.1 * embedding_weights[2] +
                           0.0 * embedding_weights[1] +
                           0.0 * embedding_weights[0])  # this one is 0 because of masking
        numpy.testing.assert_almost_equal(aggregated_tensor, [expected_tensor], decimal=5)
