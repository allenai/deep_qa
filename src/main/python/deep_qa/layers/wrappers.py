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


class FixedTimeDistributed(TimeDistributed):
    """
    This class fixes a bug in Keras where the input mask is not passed to the wrapped layer.
    """
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


class TimeDistributedWithMask(FixedTimeDistributed):
    """
    Keras' TimeDistributed layer does not call the wrapped layer in `compute_mask()`.  This
    handles the mask computation exactly as the computation of `x` in `call()`: for each time step,
    we call the wrapped layer to compute the mask for that portion of the input, then aggregate
    them.
    """
    def compute_mask(self, x, mask=None):
        if mask is None:
            return None
        input_shape = self.input_spec[0].shape
        input_length = input_shape[1] if input_shape[1] else K.shape(x)[1]
        reshaped_x = K.reshape(x, (-1,) + input_shape[2:])  # (batch_size * timesteps, ...)
        mask_ndim = K.ndim(mask)
        input_ndim = K.ndim(x)
        if mask_ndim == input_ndim:
            mask_shape = input_shape
        elif mask_ndim == input_ndim - 1:
            mask_shape = input_shape[:-1]
        else:
            raise Exception("Mask is of an unexpected shape. Mask's ndim: %s, input's ndim %s" %
                            (mask_ndim, input_ndim))
        reshaped_mask = K.reshape(mask, (-1,) + mask_shape[2:])  # (batch_size * timesteps, ...)
        output_mask = self.layer.compute_mask(reshaped_x, mask=reshaped_mask)
        if output_mask is None:
            return None
        output_mask_shape = self.layer.get_output_mask_shape_for((input_shape[0],) + input_shape[2:])
        reshaped_shape = (-1, input_length) + output_mask_shape[1:]
        outputs = K.reshape(output_mask, reshaped_shape)
        return outputs

class EncoderWrapper(FixedTimeDistributed):
    '''
    This class TimeDistributes a sentence encoder, applying the encoder to several word sequences.
    The only difference between this and the regular TimeDistributed is in how we handle the mask.
    Typically, an encoder will handle masked embedded input, and return None as its mask, as it
    just returns a vector and no more masking is necessary.  However, if the encoder is
    TimeDistributed, we might run into a situation where _all_ of the words in a given sequence are
    masked (because we padded the number of sentences, for instance).  In this case, we just want
    to mask the entire sequence.  EncoderWrapper returns a mask with the same dimension as the
    input sequences, where sequences are masked if _all_ of their words were masked.
    '''
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
