from keras.layers import Layer


class OutputMask(Layer):
    """
    This Layer is purely for debugging.  You can wrap this on a layer's output to get the mask
    output by that layer as a model output, for easier visualization of what the model is actually
    doing.

    Don't try to use this in an actual model.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(OutputMask, self).__init__(**kwargs)

    def call(self, x, mask=None):  # pylint: disable=unused-argument
        return mask
