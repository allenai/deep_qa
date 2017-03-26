from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...tensors.backend import switch, very_negative_like


class Max(MaskedLayer):
    """
    This ``Layer`` performs a max over some dimension.  Keras has a similar layer called
    ``GlobalMaxPooling1D``, but it is not as configurable as this one, and it does not support
    masking.

    If the mask is not ``None``, it must be the same shape as the input.

    Input:
        - A tensor of arbitrary shape (having at least 3 dimensions).

    Output:
        - A tensor with one less dimension, where we have taken a max over one of the dimensions.
    """
    def __init__(self, axis: int=-1, **kwargs):
        self.axis = axis
        super(Max, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.any(mask, axis=self.axis)

    @overrides
    def compute_output_shape(self, input_shape):
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)
        return input_shape[:axis] + input_shape[axis+1:]

    @overrides
    def call(self, inputs, mask=None):
        if mask is not None:
            inputs = switch(mask, inputs, very_negative_like(inputs))
        return K.max(inputs, axis=self.axis)

    @overrides
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Max, self).get_config()
        config.update(base_config)
        return config
