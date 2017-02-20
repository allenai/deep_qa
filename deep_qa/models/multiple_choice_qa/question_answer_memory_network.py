from typing import Dict, Any

from overrides import overrides
from keras.layers import Input

from ...data.instances.question_answer_instance import QuestionAnswerInstance
from ...layers.wrappers.encoder_wrapper import EncoderWrapper
from ..memory_networks.memory_network import MemoryNetwork


class QuestionAnswerMemoryNetwork(MemoryNetwork):
    '''
    This is a MemoryNetwork that is trained on QuestionAnswerInstances.

    The base MemoryNetwork assumes we're dealing with TrueFalseInstances, so there is no
    separate question and answer text.  We need to add an additional input, and handle it specially
    at the end, with a very different final "entailment" model, where we're doing dot-product
    similarity with encoded answer options, or something similar.

    Parameters
    ----------
    answer_encoder_name : string, optional (default="answer")
        Encoder names specify sharing between different encoders.  If you want to share an encoder
        between answer options and questions, be sure that the answer_encoder_name matches the
        encoder used for questions (in this case, that's currently ``"default"``).
    '''

    def __init__(self, params: Dict[str, Any]):
        self.answer_encoder_name = params.pop("answer_encoder_name", "answer")
        super(QuestionAnswerMemoryNetwork, self).__init__(params)

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
        max_lengths = super(QuestionAnswerMemoryNetwork, self)._get_max_lengths()
        max_lengths['answer_length'] = self.max_answer_length
        max_lengths['num_options'] = self.num_options
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        super(QuestionAnswerMemoryNetwork, self)._set_max_lengths(max_lengths)
        self.max_answer_length = max_lengths['answer_length']
        self.num_options = max_lengths['num_options']

    @overrides
    def _get_entailment_output(self, combined_input):
        answer_input = Input(shape=(self.num_options, self.max_answer_length), dtype='int32', name="answer_input")
        answer_embedding = self._embed_input(answer_input)
        answer_encoder = self._get_encoder(name=self.answer_encoder_name,
                                           fallback_behavior="use default encoder")
        answer_encoder = EncoderWrapper(answer_encoder, name=answer_encoder.name + "_wrapper")
        encoded_answers = answer_encoder(answer_embedding)
        return ([answer_input], self._get_entailment_model().classify(combined_input, encoded_answers))
