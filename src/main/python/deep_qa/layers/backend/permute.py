from typing import Tuple

from keras import backend as K
from keras.layers import Layer
from overrides import overrides


class Permute(Layer):
    """
    This `Layer` calls `K.permute_dimensions` on both the input and the mask.

    If the mask is not `None`, it must have the same shape as the input.

    Input:
        - A tensor of arbitrary shape.

    Output:
        - A tensor with permuted dimensions.
    """
    def __init__(self, pattern: Tuple[int], **kwargs):
        self.supports_masking = True
        self.pattern = pattern
        super(Permute, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.permute_dimensions(mask, self.pattern)

    @overrides
    def get_output_shape_for(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    @overrides
    def call(self, x, mask=None):
        return K.permute_dimensions(x, pattern=self.pattern)
