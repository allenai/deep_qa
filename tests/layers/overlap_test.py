# pylint: disable=no-self-use
import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K
from keras.layers import Input
from keras.models import Model

from deep_qa.layers import Overlap


class TestOverlap:
    def test_batched_case(self):
        tensor_a_len = 5
        tensor_b_len = 4

        tensor_a_input = Input(shape=(tensor_a_len,),
                               dtype='int32',
                               name="tensor_a")
        tensor_b_input = Input(shape=(tensor_b_len,),
                               dtype='int32',
                               name="tensor_b")
        overlap_output = Overlap()([tensor_a_input,
                                    tensor_b_input])
        model = Model([tensor_a_input,
                       tensor_b_input],
                      overlap_output)

        tensor_a = numpy.array([[1, 3, 4, 8, 2], [2, 8, 1, 2, 3]])
        tensor_b = numpy.array([[9, 4, 2, 5], [6, 1, 2, 2]])
        expected_output = numpy.array([[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
                                       [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]])

        # Testing the general batched case
        result = model.predict([tensor_a, tensor_b])
        assert_almost_equal(result, expected_output)

    def test_masked_batched_case(self):
        tensor_a = K.variable(numpy.array([[1, 3, 4, 8, 2], [2, 8, 1, 2, 3]]),
                              dtype="int32")
        tensor_b = K.variable(numpy.array([[9, 4, 2, 5], [6, 1, 2, 2]]),
                              dtype="int32")
        mask_a = K.variable(numpy.array([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]]))
        mask_b = K.variable(numpy.array([[1, 1, 0, 0], [1, 1, 0, 0]]))
        expected_output = numpy.array([[[1.0, 0.0], [1.0, 0.0],
                                        [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]],
                                       [[1.0, 0.0], [1.0, 0.0],
                                        [0.0, 1.0], [1.0, 0.0], [1.0, 0.0]]])

        # Testing the masked general batched case
        result = K.eval(Overlap()([tensor_a, tensor_b], mask=[mask_a, mask_b]))
        assert_almost_equal(result, expected_output)
