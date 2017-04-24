from keras import backend as K
from keras.engine import InputSpec
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...tensors.backend import switch



class PositionalEncoder(MaskedLayer):
    '''
    A ``PositionalEncoder`` is very similar to a kind of weighted bag of words encoder,
    where the weighting is done by an index-dependent vector, not a scalar. If you think
    this is an odd thing to do, it is. The original authors provide no real reasoning behind
    the exact method other than it takes into account word order. This is here mainly to reproduce
    results for comparison.

    It takes a matrix of shape (num_words, word_dim) and returns a vector of size (word_dim),
    which implements the following linear combination of the rows:

    representation = sum_(j=1)^(n) { l_j * w_j }

    where w_j is the j-th word representation in the sentence and l_j is a vector defined as follows:

    l_kj =  (1 - j)/m  -  (k/d)((1-2j)/m)

    where:
     - j is the word sentence index.
     - m is the sentence length.
     - k is the vector index(ie the k-th element of a vector).
     - d is the dimension of the embedding.
     - * represents element-wise multiplication.

    This method was originally introduced in End-To-End Memory Networks(pg 4-5):
    https://arxiv.org/pdf/1503.08895v5.pdf
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        self.input_spec = [InputSpec(ndim=3)]
        # For consistency of handling sentence encoders, we will often get passed this parameter.
        # We don't use it, but Layer will complain if it's there, so we get rid of it here.
        kwargs.pop('units', None)

        super(PositionalEncoder, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])  # removing second dimension

    @overrides
    def call(self, inputs, mask=None):
        # pylint: disable=redefined-variable-type

        # This section implements the positional encoder on all the vectors at once.
        # The general idea is to use ones matrices in the shape of `inputs` to create indexes per
        # word.

        if mask is None:
            ones_like_x = K.ones_like(inputs)
        else:
            float_mask = K.cast(mask, 'float32')
            ones_like_x = K.ones_like(inputs) * K.expand_dims(float_mask, 2)

        # This is an odd way to get the number of words(ie the first dimension of inputs).
        # However, if the input is masked, using the dimension directly does not
        # equate to the correct number of words. We fix this by adding up a relevant
        # row of ones which has been masked if required.
        masked_m = K.expand_dims(K.sum(ones_like_x, 1), 1)

        if mask is None:
            one_over_m = ones_like_x / masked_m
            j_index = K.cumsum(ones_like_x, 1)
        else:
            one_over_m = switch(ones_like_x, ones_like_x/masked_m, K.zeros_like(ones_like_x))

            j_index = K.cumsum(ones_like_x, 1) * K.expand_dims(float_mask, 2)

        k_over_d = K.cumsum(ones_like_x, 2) * 1.0/K.cast(K.shape(inputs)[2], 'float32')

        l_weighting_vectors = (ones_like_x - (j_index * one_over_m)) - \
                              (k_over_d * (ones_like_x - 2 * j_index * one_over_m))

        return K.sum(l_weighting_vectors * inputs, 1)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # We need to override this method because Layer passes the input mask unchanged since this
        # layer supports masking. We don't want that. After the input is merged we can stop
        # propagating the mask.
        return None
