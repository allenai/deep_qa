from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...tensors.backend import last_dim_flatten
from ...tensors.masked_operations import masked_softmax

class MaskedSoftmax(MaskedLayer):
    '''
    This Layer performs a masked softmax.  This could just be a `Lambda` layer that calls our
    `tensors.masked_softmax` function, except that `Lambda` layers do not properly handle masked
    input.

    The expected input to this layer is a tensor of shape `(batch_size, num_options)`, with a mask
    of the same shape.  We also accept an input tensor of shape `(batch_size, num_options, 1)`,
    which we will squeeze to be `(batch_size, num_options)` (though the mask must still be
    `(batch_size, num_options)`).

    While we give the expected input as having two modes, we also accept higher-order tensors.  In
    those cases, we'll first perform a `last_dim_flatten` on both the input and the mask, so that
    we always do the softmax over a single dimension (the last one).

    We give no output mask, as we expect this to only be used at the end of the model, to get a
    final probability distribution over class labels (and it's a softmax, so you'll have zeros in
    the tensor itself; do you really still need a mask?).  If you need this to propagate the mask
    for whatever reason, it would be pretty easy to change it to optionally do so - submit a PR.
    '''
    def __init__(self, **kwargs):
        super(MaskedSoftmax, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    @overrides
    def compute_output_shape(self, input_shape):
        if input_shape[-1] == 1:
            return input_shape[:-1]
        else:
            return input_shape

    @overrides
    def call(self, inputs, mask=None):
        input_shape = K.int_shape(inputs)
        if input_shape[-1] == 1:
            inputs = K.squeeze(inputs, axis=-1)
            input_shape = input_shape[:-1]
        if len(input_shape) > 2:
            inputs = last_dim_flatten(inputs)
            if mask is not None:
                mask = last_dim_flatten(mask)
        # Now we have both inputs and mask with shape (?, num_options), and can do a softmax.
        softmax_result = masked_softmax(inputs, mask)
        if len(input_shape) > 2:
            input_shape = (-1,) + input_shape[1:]
            softmax_result = K.reshape(softmax_result, input_shape)
        return softmax_result
