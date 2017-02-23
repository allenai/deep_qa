from keras import backend as K
from keras.layers import Layer
from overrides import overrides


class Squeeze(Layer):
    """
    This `Layer` removes a 1-D dimension from the tensor at index "axis", acting as simply
    a layer version of the backend squeeze function.

    If the mask is not `None`, it must be the same shape as the input.

    Input:
        - A tensor of arbitrary shape (having at least 3 dimensions).

    Output:
        - A tensor with the same data as `x` but reduced dimensions.
    """
    def __init__(self, axis: int=-1, **kwargs):
        self.supports_masking = True
        self.axis = axis
        super(Squeeze, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.squeeze(mask, axis=self.axis)

    @overrides
    def get_output_shape_for(self, input_shape):
        axis = self.axis
        if axis < 0:
            axis += len(input_shape)
        return input_shape[:axis] + input_shape[axis+1:]

    @overrides
    def call(self, x, mask=None):
        return K.squeeze(x, axis=self.axis)

    def get_config(self):
        base_config = super(Squeeze, self).get_config()
        config = {'axis': self.axis}
        config.update(base_config)
        return config
