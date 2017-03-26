from keras import backend as K
from overrides import overrides

from .masked_layer import MaskedLayer
from ..tensors.backend import switch


class BiGRUIndexSelector(MaskedLayer):
    """
    This Layer takes 3 inputs: a tensor of document indices, the seq2seq GRU output
    over the document feeding it in forward, the seq2seq GRU output over the document
    feeding it in backwards. It also takes one parameter, the word index whose
    biGRU outputs we want to extract

    Inputs:
        - document indices: shape ``(batch_size, document_length)``
        - forward GRU output: shape ``(batch_size, document_length, GRU hidden dim)``
        - backward GRU output: shape ``(batch_size, document_length, GRU hidden dim)``

    Output:
        - GRU outputs at index: shape ``(batch_size, GRU hidden dim * 2)``

    Parameters
    ----------
    target_index : int
        The word index to extract the forward and backward GRU output from.
    """
    def __init__(self, target_index, **kwargs):
        self.target_index = target_index
        super(BiGRUIndexSelector, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][2]*2)

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        return None

    @overrides
    def call(self, inputs, mask=None):
        """
        Extract the GRU output for the target document index for the forward
        and backwards GRU outputs, and then concatenate them. If the target word index
        is at index l, and there are T total document words, the desired output
        in the forward pass is at GRU_f[l] (ignoring the batched case) and the
        desired output of the backwards pass is at GRU_b[T-l].

        We need to get these two vectors and concatenate them. To do so, we'll
        reverse the backwards GRU, which allows us to use the same index/mask for both.
        """
        # TODO(nelson): deal with case where cloze token appears multiple times
        # in a question.
        word_indices, gru_f, gru_b = inputs
        index_mask = K.cast(K.equal((K.ones_like(word_indices) * self.target_index),
                                    word_indices), "float32")
        gru_mask = K.repeat_elements(K.expand_dims(index_mask, -1), K.int_shape(gru_f)[-1], K.ndim(gru_f) - 1)
        masked_gru_f = switch(gru_mask, gru_f, K.zeros_like(gru_f))
        selected_gru_f = K.sum(masked_gru_f, axis=1)
        masked_gru_b = switch(gru_mask, gru_b, K.zeros_like(gru_b))
        selected_gru_b = K.sum(masked_gru_b, axis=1)
        selected_bigru = K.concatenate([selected_gru_f, selected_gru_b], axis=-1)
        return selected_bigru

    @overrides
    def get_config(self):
        config = {'target_index': self.target_index}
        base_config = super(BiGRUIndexSelector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
