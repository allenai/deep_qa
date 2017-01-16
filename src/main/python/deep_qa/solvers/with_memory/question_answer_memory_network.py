from typing import Dict, Any

from overrides import overrides
from keras.layers import Input

from ...data.instances.question_answer_instance import QuestionAnswerInstance
from ...layers.wrappers import EncoderWrapper
from .memory_network import MemoryNetworkSolver


class QuestionAnswerMemoryNetworkSolver(MemoryNetworkSolver):
    '''
    This is a MemoryNetworkSolver that is trained on QuestionAnswerInstances.

    The base MemoryNetworkSolver assumes we're dealing with TrueFalseInstances, so there is no
    separate question and answer text.  We need to add an additional input, and handle it specially
    at the end, with a very different final "entailment" model, where we're doing dot-product
    similarity with encoded answer options, or something similar.
    '''

    def __init__(self, params: Dict[str, Any]):
        # We don't have any parameters to set that are specific to this class, so we just call the
        # superclass constructor.
        super(QuestionAnswerMemoryNetworkSolver, self).__init__(params)

        # Now we set some class-specific member variables.
        self.entailment_choices = ['question_answer_mlp']

        # And declare some model-specific variables that will be set later.
        self.num_options = None
        self.max_answer_length = None

    @overrides
    def _instance_type(self):
        return QuestionAnswerInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        max_lengths = super(QuestionAnswerMemoryNetworkSolver, self)._get_max_lengths()
        max_lengths['answer_length'] = self.max_answer_length
        max_lengths['num_options'] = self.num_options
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        super(QuestionAnswerMemoryNetworkSolver, self)._set_max_lengths(max_lengths)
        self.max_answer_length = max_lengths['answer_length']
        self.num_options = max_lengths['num_options']

    @overrides
    def _get_entailment_output(self, combined_input):
        # TODO(matt): allow this to have a separate embedding from the main embedding, which allows
        # for a closer re-implementation of prior memory networks.
        answer_input = Input(shape=(self.num_options, self.max_answer_length), dtype='int32', name="answer_input")
        answer_embedding = self._embed_input(answer_input)
        answer_encoder = EncoderWrapper(self._get_new_sentence_encoder(), name="answer_encoder")
        encoded_answers = answer_encoder(answer_embedding)
        return ([answer_input], self._get_entailment_model().classify(combined_input, encoded_answers))
