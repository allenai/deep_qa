from typing import Any, Dict
from overrides import overrides

from keras.layers import Input
from keras.engine import Layer

from .memory_network import MemoryNetworkSolver
from ...training.models import DeepQaModel


#TODO(pradeep): Properly merge this with memory networks.
class DecomposableAttentionSolver(MemoryNetworkSolver):
    '''
    This is a solver that embeds the question-option pair and the background and applies the decomposable
    attention entailment model (Parikh et al., EMNLP 2016). This currently only works with backgrounds of
    length=1 for now.

    While this class inherits MemoryNetworkSolver for now, it is only the data
    preparation and embedding steps from that class that are reused here.
    '''
    def __init__(self, params: Dict[str, Any]):
        super(DecomposableAttentionSolver, self).__init__(params)
        # This solver works only with decomposable_attention. Overwriting entailment_choices
        self.entailment_choices = ['decomposable_attention']

    @overrides
    def _build_model(self):
        question_input = Input(shape=self._get_question_shape(), dtype='int32', name="sentence_input")
        knowledge_input = Input(shape=self._get_background_shape(), dtype='int32', name="background_input")
        question_embedding = self._embed_input(question_input)
        knowledge_embedding = self._embed_input(knowledge_input)
        knowledge_embedding = TopKnowledgeSelector()(knowledge_embedding)

        entailment_layer = self._get_entailment_model()
        true_false_probabilities = entailment_layer([knowledge_embedding, question_embedding])
        return DeepQaModel(input=[question_input, knowledge_input], output=true_false_probabilities)


class TopKnowledgeSelector(Layer):
    '''
    Takes the embedding of (masked) knowledge, and returns the embedding of just the first sentence in knowledge.
    We need this because DecomposableAttentionSolver works with only one premise for now. We also assume here that
    the sentences in knowledge are sorted by their relevance.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(TopKnowledgeSelector, self).__init__(**kwargs)

    def compute_mask(self, x, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        else:
            # input mask is of shape (batch_size, knowledge_length, sentence_length)
            return mask[:, 0, :]  #(batch_size, sentence_length)

    def get_output_shape_for(self, input_shape):
        # input_shape in (batch_size, knowledge_length, sentence_length, embed_dim)
        return (input_shape[0], input_shape[2], input_shape[3])

    def call(self, x, mask=None):
        return x[:, 0, :, :]  # (batch_size, sentence_length, embed_dim)
