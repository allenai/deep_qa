from keras.layers import Layer, TimeDistributed
from keras import backend as K


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


class EncoderWrapper(TimeDistributed):
    '''
    This wrapper timedistributes an encoder. We need a separate class for doing this instead of using
    TimeDistributed itself because TimeDistributed does not handle masking the way we want:
    1) TimeDistributed does not pass the input mask to the wrapped layer as of Keras 1.1.0. We fix that here,
        but this fix may be unnecessary with a later version of Keras.
    2) TimeDistributed passes along its input mask to the output. This does not work when the wrapped layer
        outputs a tensor with different number of dimensions.
    '''
    def call(self, x, mask=None):
        # This is copied from the current implementation of call in TimeDistributed, except that this actually
        # uses the mask (and has better variable names).
        input_shape = self.input_spec[0].shape
        if input_shape[0]:
            # The batch size is passed when defining the layer in some cases (for example if it is stateful).
            # We respect the input shape in that case and not reshape the input. This is slower.
            # pylint: disable=unused-argument
            def step(x_i, states):
                output = self.layer.call(x_i)
                return output, []
            _, outputs, _ = K.rnn(step, x, mask=mask, input_states=[])
        else:
            input_length = input_shape[1] if input_shape[1] else K.shape(x)[1]
            reshaped_x = K.reshape(x, (-1,) + input_shape[2:])  # (batch_size * timesteps, ...)
            if mask is not None:
                mask_ndim = K.ndim(mask)
                input_ndim = K.ndim(x)
                if mask_ndim == input_ndim:
                    mask_shape = input_shape
                elif mask_ndim == input_ndim - 1:
                    mask_shape = input_shape[:-1]
                else:
                    raise Exception("Mask is of an unexpected shape. Mask's ndim: %s, input's ndim %s" %
                                    (mask_ndim, input_ndim))
                mask = K.reshape(mask, (-1,) + mask_shape[2:])  # (batch_size * timesteps, ...)
            outputs = self.layer.call(reshaped_x, mask=mask)
            output_shape = self.get_output_shape_for(input_shape)
            outputs = K.reshape(outputs, (-1, input_length) + output_shape[2:])
        return outputs

    def compute_mask(self, x, mask=None):
        # pylint: disable=unused-argument
        # Input mask (coming from Embedding) will be of shape (batch_size, knowledge_length, num_words).
        # Output mask should be of shape (batch_size, knowledge_length) with 0s for background sentences that
        #       are all padding.
        if mask is None:
            return None
        else:
            # An output bit is 0 only if the  bits corresponding to all input words are 0.
            return K.any(mask, axis=-1)
