from overrides import overrides

from .masked_layer import MaskedLayer


class TopKnowledgeSelector(MaskedLayer):
    '''
    Takes the embedding of (masked) knowledge, and returns the embedding of just the first sentence
    in knowledge.  We need this because DecomposableAttentionEntailment works with only one premise
    for now. We also assume here that the sentences in knowledge are sorted by their relevance.
    '''
    def __init__(self, **kwargs):
        super(TopKnowledgeSelector, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        else:
            # input mask is of shape (batch_size, knowledge_length, sentence_length)
            return mask[:, 0, :]  #(batch_size, sentence_length)

    def get_output_mask_shape_for(self, input_shape):  # pylint: disable=no-self-use
        """
        This is a method I added in order to allow for proper mask computation in TimeDistributed.
        It is called by TimeDistributedWithMask.compute_mask() - see that code for some more
        insight as to why this is necessary.

        Once we're confident that this works, I plan on submitting a pull request to Keras with our
        improved TimeDistributed class.
        """
        # input_shape is (batch_size, knowledge_length, sentence_length, embed_dim)
        mask_shape = (input_shape[0], input_shape[2])
        return mask_shape

    @overrides
    def compute_output_shape(self, input_shape):
        # input_shape is (batch_size, knowledge_length, sentence_length, embed_dim)
        return (input_shape[0], input_shape[2], input_shape[3])

    @overrides
    def call(self, inputs, mask=None):
        return inputs[:, 0, :, :]  # (batch_size, sentence_length, embed_dim)
