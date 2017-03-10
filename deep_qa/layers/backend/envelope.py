from keras.layers import Layer
from overrides import overrides

from ...tensors.backend import cumulative_sum


class Envelope(Layer):
    """
    Given a probability distribution over a begin index and an end index of some sequence, this
    ``Layer`` computes an envelope over the sequence, a probability that each element lies within
    "begin" and "end".

    Specifically, the computation done here is the following::

        after_span_begin = cumulative_sum(span_begin)
        after_span_end = cumulative_sum(span_end)
        before_span_end = 1 - after_span_end
        envelope = after_span_begin * before_span_end

    Inputs:
        - span_begin: tensor with shape ``(batch_size, sequence_length)``, representing a
          probability distribution over a start index in the sequence
        - span_end: tensor with shape ``(batch_size, sequence_length)``, representing a probability
          distribution over an end index in the sequence

    Outputs:
        - envelope: tensor with shape ``(batch_size, sequence_length)``, representing a probability
          for each index of the sequence belonging in the span

    If there is a mask associated with either of the inputs, we ignore it, assuming that you used
    the mask correctly when you computed your probability distributions.  But we support masking in
    this layer, so that you have an output mask if you really need it.  We just return the first
    mask that is not ``None`` (or ``None``, if both are ``None``).

    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Envelope, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        span_begin_mask, span_end_mask = mask
        return span_begin_mask if span_begin_mask is not None else span_end_mask

    @overrides
    def get_output_shape_for(self, input_shape):
        span_begin_shape, _ = input_shape
        return span_begin_shape

    @overrides
    def call(self, x, mask=None):
        span_begin, span_end = x
        after_span_begin = cumulative_sum(span_begin)
        after_span_end = cumulative_sum(span_end)
        before_span_end = 1.0 - after_span_end
        return after_span_begin * before_span_end
