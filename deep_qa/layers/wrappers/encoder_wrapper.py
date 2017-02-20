from keras import backend as K
from deep_qa.layers.wrappers.time_distributed import TimeDistributed


class EncoderWrapper(TimeDistributed):
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
    def compute_mask(self, x, input_mask=None):
        # pylint: disable=unused-argument
        # Input mask (coming from Embedding) will be of shape (batch_size, knowledge_length, num_words).
        # Output mask should be of shape (batch_size, knowledge_length) with 0s for background sentences that
        #       are all padding.
        if input_mask is None:
            return None
        else:
            # An output bit is 0 only if the  bits corresponding to all input words are 0.
            return K.any(input_mask, axis=-1)
