from keras import backend as K
from overrides import overrides

from ...tensors.backend import switch
from ..masked_layer import MaskedLayer


class ReplaceMaskedValues(MaskedLayer):
    """
    This ``Layer`` replaces all masked values in a tensor with some value.  You might want to do
    this before passing the tensor into a layer that does a max or a min, for example, to replace
    all masked values with something very large or very negative.  We basically just call
    ``switch`` on the mask.

    Input:
        - tensor: a tensor of arbitrary shape

    Output:
        - the same tensor, with masked values replaced by some input value

    Parameters
    ----------
    replace_with: float
        We will replace all masked values in the tensor with this value.
    """
    def __init__(self, replace_with: float, **kwargs):
        self.replace_with = replace_with
        super(ReplaceMaskedValues, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        return mask

    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape

    @overrides
    def call(self, inputs, mask=None):
        if mask is None:
            # It turns out that Keras doesn't like it if you just return inputs, so we need to
            # return a different tensor object.  Just doing a cast apparently doesn't work, either,
            # so we'll add 0.
            return inputs + 0.0
        return switch(mask, inputs, K.ones_like(inputs) * self.replace_with)

    @overrides
    def get_config(self):
        config = {'replace_with': self.replace_with}
        base_config = super(ReplaceMaskedValues, self).get_config()
        config.update(base_config)
        return config
