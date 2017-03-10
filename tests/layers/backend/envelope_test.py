# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.backend.envelope import Envelope

class TestEnvelopeLayer:
    def test_call_works_on_simple_input(self):
        batch_size = 1
        sequence_length = 5
        span_begin_input = Input(shape=(sequence_length,), dtype='float32')
        span_end_input = Input(shape=(sequence_length,), dtype='float32')
        envelope = Envelope()([span_begin_input, span_end_input])
        model = Model(input=[span_begin_input, span_end_input], output=[envelope])
        span_begin_tensor = numpy.asarray([[0.01, 0.1, 0.8, 0.05, 0.04]])
        span_end_tensor = numpy.asarray([[0.01, 0.04, 0.05, 0.2, 0.7]])
        envelope_tensor = model.predict([span_begin_tensor, span_end_tensor])
        assert envelope_tensor.shape == (batch_size, sequence_length)
        expected_envelope = [[0.01 * 0.99, 0.11 * 0.95, 0.91 * 0.9, 0.96 * 0.7, 1.0 * 0.0]]
        numpy.testing.assert_almost_equal(envelope_tensor, expected_envelope)
