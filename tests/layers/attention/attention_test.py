# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
from keras.layers import Embedding, Input
from keras.models import Model
import keras.backend as K
from deep_qa.layers.attention import Attention


class TestAttentionLayer:
    def test_no_mask(self):
        vector_length = 3
        matrix_num_rows = 2

        vector_input = Input(shape=(vector_length,),
                             dtype='float32',
                             name="vector_input")
        matrix_input = Input(shape=(matrix_num_rows, vector_length),
                             dtype='float32',
                             name="matrix_input")
        similarity_softmax = Attention()([vector_input, matrix_input])
        model = Model([vector_input, matrix_input],
                      similarity_softmax)

        # Testing general non-batched case.
        vector = numpy.array([[0.3, 0.1, 0.5]])
        matrix = numpy.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]])

        result = model.predict([vector, matrix])
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162]]))

        # Testing non-batched case where inputs are all 0s.
        vector = numpy.array([[0, 0, 0]])
        matrix = numpy.array([[[0, 0, 0], [0, 0, 0]]])

        result = model.predict([vector, matrix])
        assert_almost_equal(result, numpy.array([[0.5, 0.5]]))

    def test_masked(self):
        # Testing general masked non-batched case.
        vector = K.variable(numpy.array([[0.3, 0.1, 0.5]]))
        matrix = K.variable(numpy.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.1, 0.4, 0.3]]]))
        mask = K.variable(numpy.array([[1.0, 0.0, 1.0]]))
        result = K.eval(Attention().call([vector, matrix], mask=["_", mask]))
        assert_almost_equal(result, numpy.array([[0.52248482, 0.0, 0.47751518]]))

    def test_batched_no_mask(self):
        vector_length = 3
        matrix_num_rows = 2

        vector_input = Input(shape=(vector_length,),
                             dtype='float32',
                             name="vector_input")
        matrix_input = Input(shape=(matrix_num_rows, vector_length),
                             dtype='float32',
                             name="matrix_input")
        similarity_softmax = Attention()([vector_input, matrix_input])
        model = Model([vector_input, matrix_input],
                      similarity_softmax)

        # Testing general batched case.
        vector = numpy.array([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]])
        matrix = numpy.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]],
                              [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2]]])

        result = model.predict([vector, matrix])
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162],
                                                 [0.52871835, 0.47128162]]))

    def test_batched_masked(self):
        # Testing general masked non-batched case.
        vector = K.variable(numpy.array([[0.3, 0.1, 0.5], [0.3, 0.1, 0.5]]))
        matrix = K.variable(numpy.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                         [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = K.variable(numpy.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0]]))
        result = K.eval(Attention().call([vector, matrix], mask=["_", mask]))
        assert_almost_equal(result, numpy.array([[0.52871835, 0.47128162, 0.0],
                                                 [0.50749944, 0.0, 0.49250056]]))

        # Test the case where a mask is all 0s and an input is all 0s.
        vector = K.variable(numpy.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.5]]))
        matrix = K.variable(numpy.array([[[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]],
                                         [[0.6, 0.8, 0.1], [0.15, 0.5, 0.2], [0.5, 0.3, 0.2]]]))
        mask = K.variable(numpy.array([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]))
        result = K.eval(Attention().call([vector, matrix], mask=["_", mask]))
        assert_almost_equal(result, numpy.array([[0.5, 0.5, 0.0],
                                                 [0.0, 0.0, 0.0]]))

    def test_call_works_on_simple_input(self):
        sentence_length = 2
        embedding_dim = 3
        sentence_embedding = Input(shape=(sentence_length, embedding_dim), dtype='float32')
        query_input = Input(shape=(embedding_dim,), dtype='float32')
        attention_layer = Attention()
        attention = attention_layer([query_input, sentence_embedding])
        model = Model(inputs=[query_input, sentence_embedding], outputs=[attention])
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
        model = Model(inputs=[query_input, sentence_input], outputs=[attention])
        sentence_tensor = numpy.asarray([[0, 1, 0, 2]])
        query_tensor = numpy.asarray([[.1, .8, .5]])
        attention_tensor = model.predict([query_tensor, sentence_tensor])
        assert_almost_equal(attention_tensor, [[0, .73105858, 0, .26894142]])
