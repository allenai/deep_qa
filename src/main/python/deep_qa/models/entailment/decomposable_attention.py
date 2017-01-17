from typing import Any, Dict
from overrides import overrides

from keras.layers import Input

from ..memory_networks.memory_network import MemoryNetwork
from ...layers.top_knowledge_selector import TopKnowledgeSelector
from ...training.models import DeepQaModel


#TODO(matt): Remove the dependency on MemoryNetwork here - we're only using
# self._get_entailment_model(), and we could just instantiate the model we want easily enough, and
# that would make it so we don't need the TopKnowledgeSelector, anyway, and can just take a pair of
# sentences as input.  i.e., we should make this an actual entailment model, that can run as-is on
# SNLI.
class DecomposableAttention(MemoryNetwork):
    '''
    This is a solver that embeds the question-option pair and the background and applies the
    decomposable attention entailment model (Parikh et al., EMNLP 2016). This currently only works
    with backgrounds of length=1 for now.

    While this class inherits MemoryNetwork for now, it is only the data preparation and
    embedding steps from that class that are reused here.
    '''
    def __init__(self, params: Dict[str, Any]):
        super(DecomposableAttention, self).__init__(params)
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
