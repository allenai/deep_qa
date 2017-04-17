# pylint: disable=no-self-use
import numpy
from numpy.testing import assert_almost_equal

from keras.layers import Input
from keras.models import Model

from deep_qa.layers import BiGRUIndexSelector


class TestBiGRUIndexSelector():
    def test_batched_case(self):
        document_length = 5
        gru_hidden_dim = 2
        target = 8

        word_indices_input = Input(shape=(document_length,),
                                   dtype='int32',
                                   name="word_indices_input")
        gru_f_input = Input(shape=(document_length, gru_hidden_dim),
                            dtype='float32',
                            name="gru_f_input")
        gru_b_input = Input(shape=(document_length, gru_hidden_dim),
                            dtype='float32',
                            name="gru_b_input")
        index_bigru_output = BiGRUIndexSelector(target)([word_indices_input,
                                                         gru_f_input,
                                                         gru_b_input])
        model = Model([word_indices_input,
                       gru_f_input,
                       gru_b_input],
                      index_bigru_output)

        document_indices = numpy.array([[1, 3, 4, 8, 2], [2, 8, 1, 2, 3]])
        gru_f_input = numpy.array([[[0.1, 0.5], [0.3, 0.4], [0.4, 0.1], [0.9, 0.2], [0.1, 0.3]],
                                   [[0.4, 0.6], [0.7, 0.1], [0.3, 0.1], [0.9, 0.5], [0.4, 0.7]]])
        gru_b_input = numpy.array([[[0.7, 0.2], [0.9, 0.1], [0.3, 0.8], [0.2, 0.6], [0.7, 0.2]],
                                   [[0.2, 0.1], [0.3, 0.6], [0.2, 0.8], [0.3, 0.6], [0.4, 0.4]]])
        expected_output = numpy.array([[0.9, 0.2, 0.2, 0.6], [0.7, 0.1, 0.3, 0.6]])

        # Testing the general single-batch case.
        result = model.predict([document_indices, gru_f_input, gru_b_input])
        assert_almost_equal(result, expected_output)
