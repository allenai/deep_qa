from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...tensors.masked_operations import masked_batch_dot, masked_softmax


class MaxSimilaritySoftmax(MaskedLayer):
    '''
    This layer takes encoded questions and knowledge in a multiple choice
    setting and computes the similarity between each of the question embeddings
    and the background knowledge, and returns a softmax over the options.

    Inputs:

    - encoded_questions (batch_size, num_options, encoding_dim)
    - encoded_knowledge (batch_size, num_options, knowledge_length, encoding_dim)

    Output:

    - option_probabilities (batch_size, num_options)

    This is a pretty niche layer that does a very specific computation. We only
    made it its own class instead of a ``Lambda`` layer so that we could handle
    masking correctly, which ``Lambda`` does not.

    '''
    def __init__(self, knowledge_axis, max_knowledge_length, **kwargs):
        self.knowledge_axis = knowledge_axis
        self.max_knowledge_length = max_knowledge_length
        super(MaxSimilaritySoftmax, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return None

    @overrides
    def compute_output_shape(self, input_shapes):
        # (batch_size, num_options)
        return (input_shapes[0][0], input_shapes[0][1])

    @overrides
    def call(self, inputs, mask=None):
        questions, knowledge = inputs
        question_mask, knowledge_mask = mask
        question_knowledge_similarity = masked_batch_dot(questions, knowledge, question_mask, knowledge_mask)
        max_knowledge_similarity = K.max(question_knowledge_similarity, axis=-1)  # (samples, num_options)
        return masked_softmax(max_knowledge_similarity, question_mask)
