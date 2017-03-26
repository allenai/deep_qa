from typing import Tuple

from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class Permute(MaskedLayer):
    """
    This ``Layer`` calls ``K.permute_dimensions`` on both the input and the mask.

    If the mask is not ``None``, it must have the same shape as the input.

    Input:
        - A tensor of arbitrary shape.

    Output:
        - A tensor with permuted dimensions.
    """
    def __init__(self, pattern: Tuple[int], **kwargs):
        self.pattern = pattern
        super(Permute, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.permute_dimensions(mask, self.pattern)

    @overrides
    def compute_output_shape(self, input_shape):
        return tuple([input_shape[i] for i in self.pattern])

    @overrides
    def call(self, inputs, mask=None):
        return K.permute_dimensions(inputs, pattern=self.pattern)
