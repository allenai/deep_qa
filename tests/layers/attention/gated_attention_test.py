# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K
from keras.layers import Input
from keras.models import Model
from deep_qa.layers.attention import GatedAttention


class TestGatedAttentionLayer:
    def test_multiplication(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention()([document_input, question_input,
                                            attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.111, 0.068],
                                                  [0.252, 0.256],
                                                  [0.432, 0.117]]]))

    def test_masked_multiplication(self):
        # test masked batch case
        document = K.variable(numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]]))
        document_mask = K.variable(numpy.array([[1, 1, 0]]))
        question = K.variable(numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7],
                                            [0.1, .6]]]))
        attention = K.variable(numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                             [0.4, 0.2, 0.8, 0.7],
                                             [0.8, 0.1, 0.6, 0.4]]]))
        gated_attention = GatedAttention(gating_function="*")
        result = K.eval(gated_attention([document, question, attention],
                                        mask=[document_mask]))
        assert_almost_equal(result, numpy.array([[[0.111, 0.068],
                                                  [0.252, 0.256],
                                                  [0.0, 0.0]]]))

    def test_addition(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention(gating_function="+")([document_input, question_input,
                                                               attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.67, 0.78],
                                                  [1.03, 1.48],
                                                  [1.34, 1.27]]]))

    def test_masked_addition(self):
        # test masked batch case
        document = K.variable(numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]]))
        document_mask = K.variable(numpy.array([[1, 1, 0]]))
        question = K.variable(numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7],
                                            [0.1, .6]]]))
        attention = K.variable(numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                             [0.4, 0.2, 0.8, 0.7],
                                             [0.8, 0.1, 0.6, 0.4]]]))
        gated_attention = GatedAttention(gating_function="+")
        result = K.eval(gated_attention([document, question, attention],
                                        mask=[document_mask]))
        assert_almost_equal(result, numpy.array([[[0.67, 0.78],
                                                  [1.03, 1.48],
                                                  [0.0, 0.0]]]))

    def test_concatenation(self):
        document_len = 3
        question_len = 4
        bigru_hidden_dim = 2

        document_input = Input(shape=(document_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="document_input")
        question_input = Input(shape=(question_len, bigru_hidden_dim,),
                               dtype='float32',
                               name="question_input")
        attention_input = Input(shape=(document_len, question_len,),
                                dtype='float32',
                                name="attention_input")

        gated_attention = GatedAttention(gating_function="||")([document_input, question_input,
                                                                attention_input])
        model = Model([document_input, question_input, attention_input],
                      gated_attention)

        # Testing general non-batched case.
        document = numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]])
        question = numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7], [0.1, .6]]])
        attention = numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                  [0.4, 0.2, 0.8, 0.7],
                                  [0.8, 0.1, 0.6, 0.4]]])
        result = model.predict([document, question, attention])
        assert_almost_equal(result, numpy.array([[[0.37, 0.68, 0.3, 0.1],
                                                  [0.63, 1.28, 0.4, 0.2],
                                                  [0.54, 1.17, 0.8, 0.1]]]))


    def test_masked_concatenation(self):
        # test masked batch case
        document = K.variable(numpy.array([[[0.3, 0.1], [0.4, 0.2], [0.8, 0.1]]]))
        document_mask = K.variable(numpy.array([[1, 1, 0]]))
        question = K.variable(numpy.array([[[0.2, 0.6], [0.4, 0.3], [0.5, 0.7],
                                            [0.1, .6]]]))
        attention = K.variable(numpy.array([[[0.3, 0.1, 0.5, 0.2],
                                             [0.4, 0.2, 0.8, 0.7],
                                             [0.8, 0.1, 0.6, 0.4]]]))
        gated_attention = GatedAttention(gating_function="||")
        result = K.eval(gated_attention([document, question, attention],
                                        mask=[document_mask]))
        assert_almost_equal(result, numpy.array([[[0.37, 0.68, 0.3, 0.1],
                                                  [0.63, 1.28, 0.4, 0.2],
                                                  [0.0, 0.0, 0.0, 0.0]]]))
