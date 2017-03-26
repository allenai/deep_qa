from overrides import overrides

from keras import backend as K
from keras.engine import InputSpec

from ..masked_layer import MaskedLayer

class BOWEncoder(MaskedLayer):
    '''
    Bag of Words Encoder takes a matrix of shape (num_words, word_dim) and returns a vector of size (word_dim),
    which is an average of the (unmasked) rows in the input matrix. This could have been done using a Lambda
    layer, except that Lambda layer does not support masking (as of Keras 1.0.7).
    '''
    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]

        # For consistency of handling sentence encoders, we will often get passed this parameter.
        # We don't use it, but Layer will complain if it's there, so we get rid of it here.
        kwargs.pop('units', None)
        super(BOWEncoder, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # removing second dimension

    @overrides
    def call(self, inputs, mask=None):
        # pylint: disable=redefined-variable-type
        if mask is None:
            return K.mean(inputs, axis=1)
        else:
            # Compute weights such that masked elements have zero weights and the remaining
            # weight is ditributed equally among the unmasked elements.
            # Mask (samples, num_words) has 0s for masked elements and 1s everywhere else.
            # Mask is of type int8. While theano would automatically make weighted_mask below
            # of type float32 even if mask remains int8, tensorflow would complain. Let's cast it
            # explicitly to remain compatible with tf.
            float_mask = K.cast(mask, 'float32')
            # Expanding dims of the denominator to make it the same shape as the numerator, epsilon added to avoid
            # division by zero.
            # (samples, num_words)
            weighted_mask = float_mask / (K.sum(float_mask, axis=1, keepdims=True) + K.epsilon())
            if K.ndim(weighted_mask) < K.ndim(inputs):
                weighted_mask = K.expand_dims(weighted_mask)
            return K.sum(inputs * weighted_mask, axis=1)  # (samples, word_dim)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # We need to override this method because Layer passes the input mask unchanged since this layer
        # supports masking. We don't want that. After the input is averaged, we can stop propagating
        # the mask.
        return None
