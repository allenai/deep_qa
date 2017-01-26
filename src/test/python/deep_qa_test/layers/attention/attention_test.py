# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
from keras.layers import Embedding, Input
from keras.models import Model

from deep_qa.layers.attention.attention import Attention

class TestAttentionLayer:
    def test_call_works_on_simple_input(self):
        sentence_length = 2
        embedding_dim = 3
        sentence_embedding = Input(shape=(sentence_length, embedding_dim), dtype='float32')
        query_input = Input(shape=(embedding_dim,), dtype='float32')
        attention_layer = Attention()
        attention = attention_layer([query_input, sentence_embedding])
        model = Model(input=[query_input, sentence_embedding], output=[attention])
        sentence_tensor = numpy.asarray([[[1, 1, 1], [-1, 0, 1]]])
        query_tensor = numpy.asarray([[.1, .8, .5]])
        attention_tensor = model.predict([query_tensor, sentence_tensor])
        assert_almost_equal(attention_tensor, [[.73105858, .26894142]])

    def test_call_handles_masking_properly(self):
        sentence_length = 4
        vocab_size = 4
        embedding_dim = 3
        embedding_weights = numpy.asarray([[0, 0, 0], [1, 1, 1], [-1, 0, 1], [-1, -1, 0]])
        embedding = Embedding(vocab_size, embedding_dim, weights=[embedding_weights], mask_zero=True)
        sentence_input = Input(shape=(sentence_length,), dtype='int32')
        sentence_embedding = embedding(sentence_input)
        query_input = Input(shape=(embedding_dim,), dtype='float32')
        attention_layer = Attention()
        attention = attention_layer([query_input, sentence_embedding])
        model = Model(input=[query_input, sentence_input], output=[attention])
        sentence_tensor = numpy.asarray([[0, 1, 0, 2]])
        query_tensor = numpy.asarray([[.1, .8, .5]])
        attention_tensor = model.predict([query_tensor, sentence_tensor])
        assert_almost_equal(attention_tensor, [[0, .73105858, 0, .26894142]])
