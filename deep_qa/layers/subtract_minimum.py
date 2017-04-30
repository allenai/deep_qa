from keras import backend as K
from overrides import overrides

from deep_qa.layers.masked_layer import MaskedLayer
from deep_qa.tensors.backend import VERY_LARGE_NUMBER

class SubtractMinimum(MaskedLayer):
    '''
    This layer is used to normalize across a tensor axis.  Normalization is done by finding the
    minimum value across the specified axis, and then subtracting that value from all values
    (again, across the spcified axis).  Note that this also works just fine if you want to find the
    minimum across more than one axis.

    Inputs:
        - A tensor with arbitrary dimension, and a mask of the same shape (currently doesn't
          support masks with other shapes).

    Output:
        - The same tensor, with the minimum across one (or more) of the dimensions subtracted.

    Parameters
    ----------
    axis: int
        The axis (or axes) across which to find the minimum.  Can be a single int, a list of ints,
        or None.  We just call `K.min` with this parameter, so anything that's valid there works
        here too.
    '''
    def __init__(self, axis: int, **kwargs):
        self.axis = axis
        super(SubtractMinimum, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shape): # pylint: disable=no-self-use
        return input_shape

    @overrides
    def compute_mask(self, inputs, mask=None):
        return mask

    @overrides
    def call(self, inputs, mask=None):
        if mask is not None:
            mask_value = False if K.dtype(mask) == 'bool' else 0
            # Make sure masked values don't affect the input, by adding a very large number.
            mask_flipped_and_scaled = K.cast(K.equal(mask, mask_value), "float32") * VERY_LARGE_NUMBER
            minimums = K.min(inputs + mask_flipped_and_scaled, axis=self.axis, keepdims=True)
        else:
            minimums = K.min(inputs, axis=self.axis, keepdims=True)
        normalized = inputs - minimums
        return normalized

    @overrides
    def get_config(self):
        base_config = super(SubtractMinimum, self).get_config()
        config = {'axis': self.axis}
        config.update(base_config)
        return config
