from keras import backend as K
from keras.layers import Layer
from overrides import overrides

from ...common.tensors import masked_softmax

class MaskedSoftmax(Layer):
    '''
    This Layer performs a masked softmax.  This could just be a `Lambda` layer that calls our
    `tensors.masked_softmax` function, except that `Lambda` layers do not properly handle masked
    input.

    The expected input to this layer is a tensor of shape `(batch_size, num_options)`, with a mask
    of the same shape.  We also accept an input tensor of shape `(batch_size, num_options, 1)`,
    which we will squeeze to be `(batch_size, num_options)` (though the mask must still be
    `(batch_size, num_options)`).

    We give no output mask, as we expect this to only be used at the end of the model, to get a
    final probability distribution over class labels.  If you need this to propagate the mask for
    your model, it would be pretty easy to change it to optionally do so - submit a PR.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedSoftmax, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1])

    @overrides
    def call(self, x, mask=None):
        # input shape: (batch_size, num_options, 1)
        if K.ndim(x) == 3:
            x = K.squeeze(x, axis=2)
        if K.ndim(x) != 2:
            raise RuntimeError("MaskedSoftmax only supports inputs of shape (batch_size, "
                               "num_options) or (batch_size, num_options, 1)")
        return masked_softmax(x, mask)
