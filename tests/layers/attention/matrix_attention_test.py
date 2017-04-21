# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_allclose
from keras.layers import Dense, Embedding, Input
from keras.models import Model, load_model

from deep_qa.layers.attention import MatrixAttention
from deep_qa.layers.wrappers import OutputMask
from ...common.test_case import DeepQaTestCase


class TestMatrixAttentionLayer(DeepQaTestCase):
    def test_call_works_on_simple_input(self):
        sentence_1_length = 2
        sentence_2_length = 3
        embedding_dim = 3
        sentence_1_embedding = Input(shape=(sentence_1_length, embedding_dim), dtype='float32')
        sentence_2_embedding = Input(shape=(sentence_2_length, embedding_dim,), dtype='float32')
        attention_layer = MatrixAttention()
        attention = attention_layer([sentence_1_embedding, sentence_2_embedding])
        model = Model(inputs=[sentence_1_embedding, sentence_2_embedding], outputs=[attention])
        sentence_1_tensor = numpy.asarray([[[1, 1, 1], [-1, 0, 1]]])
        sentence_2_tensor = numpy.asarray([[[1, 1, 1], [-1, 0, 1], [-1, -1, -1]]])
        attention_tensor = model.predict([sentence_1_tensor, sentence_2_tensor])
        assert attention_tensor.shape == (1, sentence_1_length, sentence_2_length)
        assert_allclose(attention_tensor, [[[3, 0, -3], [0, 2, 0]]])

    def test_model_loads_correctly(self):
        sentence_1_length = 2
        sentence_2_length = 3
        embedding_dim = 3
        sentence_1_embedding = Input(shape=(sentence_1_length, embedding_dim), dtype='float32')
        sentence_2_embedding = Input(shape=(sentence_2_length, embedding_dim,), dtype='float32')
        similarity_function_params = {'type': 'linear', 'combination': 'x,y,x*y'}
        attention_layer = MatrixAttention(similarity_function=similarity_function_params)
        attention = attention_layer([sentence_1_embedding, sentence_2_embedding])
        attention = Dense(2)(attention)
        model = Model(inputs=[sentence_1_embedding, sentence_2_embedding], outputs=[attention])

        sentence_1_tensor = numpy.asarray([[[1, 1, 1], [-1, 0, 1]]])
        sentence_2_tensor = numpy.asarray([[[1, 1, 1], [-1, 0, 1], [-1, -1, -1]]])
        model_file = self.TEST_DIR + "model.tmp"
        before_loading = model.predict([sentence_1_tensor, sentence_2_tensor])

        model.save(model_file)
        model = load_model(model_file,  # pylint: disable=redefined-variable-type
                           custom_objects={'MatrixAttention': MatrixAttention})
        after_loading = model.predict([sentence_1_tensor, sentence_2_tensor])

        assert_allclose(before_loading, after_loading)

    def test_call_handles_masking_properly(self):
        sentence_length = 4
        vocab_size = 4
        embedding_dim = 3
        embedding_weights = numpy.asarray([[0, 0, 0], [1, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_weights], mask_zero=True)
        sentence_1_input = Input(shape=(sentence_length,), dtype='int32')
        sentence_2_input = Input(shape=(sentence_length,), dtype='int32')
        sentence_1_embedding = embedding(sentence_1_input)
        sentence_2_embedding = embedding(sentence_2_input)
        attention_layer = MatrixAttention()
        attention = attention_layer([sentence_1_embedding, sentence_2_embedding])
        attention_mask = OutputMask()(attention)
        model = Model(inputs=[sentence_1_input, sentence_2_input], outputs=[attention, attention_mask])
        sentence_1_tensor = numpy.asarray([[0, 0, 1, 3]])
        sentence_2_tensor = numpy.asarray([[0, 1, 0, 2]])
        attention_tensor, attention_mask = model.predict([sentence_1_tensor, sentence_2_tensor])
        expected_attention = numpy.asarray([[[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 3, 0, 0],
                                             [0, -2, 0, 1]]])
        expected_mask = numpy.asarray([[[0, 0, 0, 0],
                                        [0, 0, 0, 0],
                                        [0, 1, 0, 1],
                                        [0, 1, 0, 1]]])
        assert_allclose(attention_tensor, expected_attention)
        assert_allclose(attention_mask, expected_mask)
