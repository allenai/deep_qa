# pylint: disable=no-self-use,invalid-name

import numpy
from flaky import flaky
from keras.layers import Input
from keras.models import Model

from deep_qa.layers import ComplexConcat

class TestComplexConcatLayer:
    def test_call_works_on_simple_input(self):
        input_shape = (3, 4, 5, 7)
        input_1 = Input(shape=input_shape[1:], dtype='float32')
        input_2 = Input(shape=input_shape[1:], dtype='float32')
        input_3 = Input(shape=input_shape[1:], dtype='float32')
        input_4 = Input(shape=input_shape[1:], dtype='float32')
        inputs = [input_1, input_2, input_3, input_4]
        concatenated = ComplexConcat(combination='1,2,3,4')(inputs)
        model = Model(inputs=inputs, outputs=[concatenated])
        input_1_tensor = numpy.random.rand(*input_shape)
        input_2_tensor = numpy.random.rand(*input_shape)
        input_3_tensor = numpy.random.rand(*input_shape)
        input_4_tensor = numpy.random.rand(*input_shape)
        input_tensors = [input_1_tensor, input_2_tensor, input_3_tensor, input_4_tensor]
        concat_tensor = model.predict(input_tensors)
        assert concat_tensor.shape == (3, 4, 5, 7*4)
        numpy.testing.assert_almost_equal(concat_tensor, numpy.concatenate(input_tensors, axis=-1))

    @flaky
    def test_call_handles_complex_combinations(self):
        input_shape = (3, 4, 5, 7)
        input_1 = Input(shape=input_shape[1:], dtype='float32')
        input_2 = Input(shape=input_shape[1:], dtype='float32')
        input_3 = Input(shape=input_shape[1:], dtype='float32')
        input_4 = Input(shape=input_shape[1:], dtype='float32')
        inputs = [input_1, input_2, input_3, input_4]
        concatenated = ComplexConcat(combination='1-2,2*4,3/1,4+3,3', axis=1)(inputs)
        model = Model(inputs=inputs, outputs=[concatenated])
        input_1_tensor = numpy.random.rand(*input_shape)
        input_2_tensor = numpy.random.rand(*input_shape)
        input_3_tensor = numpy.random.rand(*input_shape)
        input_4_tensor = numpy.random.rand(*input_shape)
        input_tensors = [input_1_tensor, input_2_tensor, input_3_tensor, input_4_tensor]
        concat_tensor = model.predict(input_tensors)
        assert concat_tensor.shape == (3, 4*5, 5, 7)
        expected_tensor = numpy.concatenate([
                input_1_tensor - input_2_tensor,
                input_2_tensor * input_4_tensor,
                input_3_tensor / input_1_tensor,
                input_4_tensor + input_3_tensor,
                input_3_tensor
                ], axis=1)
        numpy.testing.assert_almost_equal(concat_tensor, expected_tensor, decimal=3)
