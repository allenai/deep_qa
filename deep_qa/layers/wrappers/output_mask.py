from overrides import overrides

from ..masked_layer import MaskedLayer


class OutputMask(MaskedLayer):
    """
    This Layer is purely for debugging.  You can wrap this on a layer's output to get the mask
    output by that layer as a model output, for easier visualization of what the model is actually
    doing.

    Don't try to use this in an actual model.
    """
    def __init__(self, **kwargs):
        # The base layer class overwrites self.supports_masking,
        # so we add it afterwards.
        super(OutputMask, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        return None

    @overrides
    def call(self, inputs, mask=None):  # pylint: disable=unused-argument
        return mask
