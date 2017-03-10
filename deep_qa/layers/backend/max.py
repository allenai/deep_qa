from keras import backend as K
from keras.layers import Layer
from overrides import overrides

from ...tensors.backend import switch, very_negative_like


class Max(Layer):
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
        self.supports_masking = True
        self.axis = axis
        super(Max, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.any(mask, axis=self.axis)

    @overrides
    def get_output_shape_for(self, input_shape):
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)
        return input_shape[:axis] + input_shape[axis+1:]

    @overrides
    def call(self, x, mask=None):
        if mask is not None:
            x = switch(mask, x, very_negative_like(x))
        return K.max(x, axis=self.axis)

    @overrides
    def get_config(self):
        config = {'axis': self.axis}
        base_config = super(Max, self).get_config()
        config.update(base_config)
        return config
