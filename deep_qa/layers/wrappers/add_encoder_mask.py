from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class AddEncoderMask(MaskedLayer):
    """
    This ``Layer`` handles masking for ``TimeDistributed`` encoders, like LSTMs, that condense
    sequences of vectors into single vectors (not LSTMs that return sequences; masking is already
    handled there correctly).  Our :class:`~.encoder_wrapper.EncoderWrapper` class does the correct
    masking computation, but it inherits from ``TimeDistributed``, which does not work with unknown
    dimensions at run-time.  If you want to wrap an encoder using
    :class:`~..backend.CollapseToBatch` and :class:`~..backend.ExpandFromBatch`, you need a way to
    get the mask back into the right form after running your encoder.  This is an issue because
    Keras' encoders don't return masks when they output single vectors.

    For example, say you have a list of sentences, like [[5, 2, 1, 0], [2, 3, 1, 1], [0, 0, 0, 0]]
    (using word indices instead of embeddings for simplicity), which has been padded to be three
    sentences, even though only two of them are actually used.  After passing it though an encoder,
    you'll have something like [[vector], [vector], [vector]], and you want a mask that looks like
    [1, 1, 0].  Keras' LSTMs and such won't give this to you.  This method adds it back.

    Inputs:
        - A tensor with shape ``(batch_size, ..., encoding_dim)`` that is the output of some
          encoder that you got with
          :func:`~deep_qa.training.text_trainer.TextTrainer._get_encoder()` (not a seq2seq encoder
          that returns sequences).
          The mask for this tensor must be ``None``.
        - A tensor with shape ``(batch_size, ..., num_words, embedding_dim)`` that was the `input`
          to that encoder.  The mask for this tensor must have shape ``(batch_size, ...,
          num_words)``.

    Output:
        - The first input tensor, with a mask computed from the second input tensor.  The
          computation is just ``K.any()`` on the last dimension.
    """
    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @overrides
    def compute_mask(self, inputs, mask=None):
        encoder_mask, embedding_mask = mask
        if encoder_mask is not None:
            raise RuntimeError("Refusing to add an encoder mask, because the tensor already has one")
        return K.any(embedding_mask, axis=-1)

    @overrides
    def call(self, inputs, mask=None):  # pylint: disable=unused-argument
        # It turns out that Keras doesn't like it if you just return inputs, so we need to return a
        # different tensor object.  Just doing a cast apparently doesn't work, either, so we'll
        # add 0.
        return inputs[0] + 0.0
