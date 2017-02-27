from typing import Dict, Any

from keras.layers import Input, Layer
from overrides import overrides

from deep_qa.data.instances.tuple_inference_instance import TupleInferenceInstance
from deep_qa.layers.tuple_matchers.word_overlap_tuple_matcher import WordOverlapTupleMatcher
from deep_qa.layers.tuple_matchers import tuple_matchers
from ...layers.attention.masked_softmax import MaskedSoftmax
from ...layers.backend.repeat import Repeat
from ...layers.backend.squeeze import Squeeze
from ...layers.noisy_or import NoisyOr
from ...layers.wrappers.time_distributed import TimeDistributed
from ...training.models import DeepQaModel
from ...training.text_trainer import TextTrainer
from ...common.params import get_choice_with_default


class TupleInferenceModel(TextTrainer):
    """
    This ``TextTrainer`` implements the TupleEntailment model of Tushar.  It takes a set of tuples
    from the question and its answer candidates and a set of background knowledge tuples and looks
    for entailment between the corresponding tuple slots.  The result is a probability distribution
    over the answer options based on how well they align with the background tuples, given the
    question text.  We consider this alignment to be a form of soft inference, hence the model
    name.

    Parameters
    ----------
    tuple_matcher: Dict[str, Any]
        Parameters for selecting and then initializing the inner entailment model, one of the
        TupleMatch models.

    noisy_or_param_init: str, default='uniform'
        The initialization for the noise parameters in the ``NoisyOr`` layers.

    num_question_tuples: int, default=10
        The number of tuples for each of the answer candidates in the question.

    num_background_tuples: int, default=10
        The number of tuples for the background knowledge.

    num_tuple_slots: int, default=4
        The number of slots in each tuple.

    num_slot_words: int, default=5
        The number of words in each slot of the tuples.

    num_options: int, default=4
        The number of answer options/candidates.

    """
    def __init__(self, params: Dict[str, Any]):
        self.noisy_or_param_init = params.pop('noisy_or_param_init', 'uniform')
        self.num_question_tuples = params.pop('num_question_tuples', 10)
        self.num_background_tuples = params.pop('num_background_tuples', 10)
        self.num_tuple_slots = params.pop('num_tuple_slots', 4)
        self.num_slot_words = params.pop('word_sequence_length', 5)
        self.num_options = params.pop('num_answer_options', 4)
        tuple_matcher_params = params.pop('tuple_matcher', {})
        tuple_matcher_choice = get_choice_with_default(tuple_matcher_params,
                                                       "type",
                                                       list(tuple_matchers.keys()))
        tuple_matcher_class = tuple_matchers[tuple_matcher_choice]
        # This is a little ugly, but necessary, because the Keras Layer API treats arguments
        # differently than our model API, and we need access to the TextTrainer object in the tuple
        # matcher if it's not a Layer.
        if issubclass(tuple_matcher_class, Layer):
            # These TimeDistributed wrappers correspond to distributing across each of num_options,
            # num_question_tuples, and num_background_tuples.
            match_layer = tuple_matcher_class(**tuple_matcher_params)
            self.tuple_matcher = TimeDistributed(TimeDistributed(TimeDistributed(match_layer)))
        else:
            self.tuple_matcher(self, tuple_matcher_params)
        super(TupleInferenceModel, self).__init__(params)

        self.name = 'TupleInferenceModel'

    @overrides
    def _instance_type(self):
        return TupleInferenceInstance

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(TupleInferenceModel, cls)._get_custom_objects()
        # TODO(matt): Figure out a better way to handle the concrete TupleMatchers here.
        custom_objects['WordOverlapTupleMatcher'] = WordOverlapTupleMatcher
        custom_objects['MaskedSoftmax'] = MaskedSoftmax
        custom_objects['NoisyOr'] = NoisyOr
        custom_objects['Repeat'] = Repeat
        custom_objects['Squeeze'] = Squeeze
        return custom_objects

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        max_lengths = super(TupleInferenceModel, self)._get_max_lengths()
        max_lengths['num_question_tuples'] = self.num_question_tuples
        max_lengths['num_background_tuples'] = self.num_background_tuples
        max_lengths['num_slots'] = self.num_tuple_slots
        max_lengths['word_sequence_length'] = self.num_slot_words
        max_lengths['num_options'] = self.num_options
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        super(TupleInferenceModel, self)._set_max_lengths(max_lengths)
        self.num_question_tuples = max_lengths['num_question_tuples']
        self.num_background_tuples = max_lengths['num_background_tuples']
        self.num_tuple_slots = max_lengths['num_slots']
        self.num_slot_words = max_lengths['word_sequence_length']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        # TODO(becky): actually implement this (it's required for loading a saved model)
        pass

    @overrides
    def _build_model(self):
        r"""
        The basic outline of the model is that the question input, :math:`\mathcal{A}` (which consists of the
        inputs for each of the answer choices, i.e., each :math:`A^c \in \mathcal{A}`), and the background input,
        :math:`\mathcal{K}`, get tiled to be the same size.  They are then aligned tuple-by-tuple: each of the
        background tuples, :math:`k_j` is compared to each of the answer tuples, :math:`a_i^c`, to create a
        support/entailment score, :math:`s_{ij}^c`.  This score is determined using the selected ``TupleMatch``
        layer.
        Then, for each answer tuple, :math:`a_i^c \in A^c` we combine
        the scores for each :math:`k_j \in K` using noisy-or to get the entailment score for the given answer
        choice tuple:
            :math:`s_i^c = 1 - \prod_{j=1:J}(1 - q_1 * s_{ij}^c)`
        where q_1 is the noise parameter for this first noisy-or.  Next, we combine these scores for each answer
        choice again using the noisy-or to get the entailment score for the answer candidate:
            :math:`s^c = 1 - \prod_{i=1:N}(1 - q_2 * s_{i}^c)`
        where q_2 is the noise parameter for this second noisy-or.  At this point, we have a score for each of
        the answer candidates, and so we perform a softmax to determine which is the best answer.
        """
        # shape: (batch size, num_options, num_question_tuples, num_tuple_slots, num_slot_words)
        slot_shape = self._get_sentence_shape(self.num_slot_words)
        question_input_shape = (self.num_options, self.num_question_tuples, self.num_tuple_slots) + slot_shape
        question_input = Input(question_input_shape, dtype='int32', name='question_input')
        # shape: (batch size, num_background_tuples, num_tuple_slots, num_slot_words)
        background_input_shape = (self.num_background_tuples, self.num_tuple_slots) + slot_shape
        background_input = Input(background_input_shape, dtype='int32', name='background_input')

        # Expand and tile the question input to be:
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples, num_tuple_slots,
        #         num_slot_words)
        tiled_question = Repeat(axis=3, repetitions=self.num_background_tuples)(question_input)

        # Expand and tile the background input to match question input.
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples, num_tuple_slots,
        #               num_slot_words)
        # First, add num_options.
        tiled_background = Repeat(axis=1, repetitions=self.num_options)(background_input)
        # Next, add num_question_tuples.
        tiled_background = Repeat(axis=2, repetitions=self.num_question_tuples)(tiled_background)

        # Find the matches between the question and background tuples.
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples, 1)
        matches = self.tuple_matcher([tiled_question, tiled_background])
        # Squeeze to get rid of the last dim of length 1.
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples)
        matches = Squeeze(axis=-1)(matches)

        # Find the probability that any given question tuple is entailed by the given background tuples.
        # shape: (batch size, num_options, num_question_tuples)
        combine_background_evidence = NoisyOr(axis=-1, param_init=self.noisy_or_param_init, name="noisy_or_1")
        qi_probabilities = combine_background_evidence(matches)

        # Find the probability that any given option is correct, given the entailement scores of each of its
        # question tuples given the set of background tuples.
        # shape: (batch size, num_options)
        combine_question_evidence = NoisyOr(axis=-1, param_init=self.noisy_or_param_init, name="noisy_or_2")
        options_probabilities = combine_question_evidence(qi_probabilities)

        # Softmax over the options to choose the best one.
        final_output = MaskedSoftmax()(options_probabilities)

        return DeepQaModel(input=[question_input, background_input], output=[final_output])
