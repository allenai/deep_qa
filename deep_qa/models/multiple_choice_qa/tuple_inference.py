from typing import Dict, List
import textwrap

from keras.layers import Input
from overrides import overrides
import numpy

from ...data.instances.multiple_choice_qa import TupleInferenceInstance
from ...layers import NoisyOr
from ...layers.attention import MaskedSoftmax
from ...layers.backend import RepeatLike
from ...layers.subtract_minimum import SubtractMinimum
from ...layers.tuple_matchers import tuple_matchers
from ...training import TextTrainer
from ...training.models import DeepQaModel
from ...common.params import Params


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

    normalize_tuples_across_answers: bool, default=False
        Whether or not to normalize each question tuple's score across the answer options.  This
        assumes that the tuples are in the same order for all answer options.  Normalization is
        currently done by subtracting the minimum score for a given tuple "position" from all the
        tuples in that position.

    display_text_wrap: int, default=150
        This is used by the debug output methods to wrap long tuple strings.

    display_num_tuples: int, default=5
        This is used by the debug output methods.  It determines how many background tuples to display for
        each of the answer tuples in a given instance when displaying the tuple match scores.

    """
    def __init__(self, params: Params):
        self.noisy_or_param_init = params.pop('noisy_or_param_init', 'uniform')
        self.num_question_tuples = params.pop('num_question_tuples', None)
        self.num_background_tuples = params.pop('num_background_tuples', None)
        self.num_tuple_slots = params.pop('num_tuple_slots', None)
        self.num_slot_words = params.pop('num_slot_words', None)
        self.num_options = params.pop('num_answer_options', None)
        self.normalize_tuples_across_answers = params.pop('normalize_tuples_across_answers', False)
        self.display_text_wrap = params.pop('display_text_wrap', 150)
        self.display_num_tuples = params.pop('display_num_tuples', 5)
        tuple_matcher_params = params.pop('tuple_matcher', {})
        tuple_matcher_choice = tuple_matcher_params.pop_choice("type", list(tuple_matchers.keys()),
                                                               default_to_first_choice=True)
        tuple_matcher_class = tuple_matchers[tuple_matcher_choice]
        self.tuple_matcher = tuple_matcher_class(self, tuple_matcher_params)
        self.tuple_matcher.name = "match_layer"
        super(TupleInferenceModel, self).__init__(params)

        self.name = 'TupleInferenceModel'

    @overrides
    def _instance_type(self):
        return TupleInferenceInstance

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(TupleInferenceModel, cls)._get_custom_objects()
        for tuple_matcher in tuple_matchers.values():
            custom_objects.update(tuple_matcher.get_custom_objects())
        custom_objects['MaskedSoftmax'] = MaskedSoftmax
        custom_objects['NoisyOr'] = NoisyOr
        custom_objects['RepeatLike'] = RepeatLike
        custom_objects['SubtractMinimum'] = SubtractMinimum
        return custom_objects

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super(TupleInferenceModel, self).get_padding_lengths()
        padding_lengths['num_question_tuples'] = self.num_question_tuples
        padding_lengths['num_background_tuples'] = self.num_background_tuples
        padding_lengths['num_slots'] = self.num_tuple_slots
        padding_lengths['num_sentence_words'] = self.num_slot_words
        padding_lengths['num_options'] = self.num_options
        return padding_lengths

    @overrides
    def get_instance_sorting_keys(self) -> List[str]:  # pylint: disable=no-self-use
        return ['num_sentence_words', 'num_background_tuples', 'num_question_tuples', 'num_slots']

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        super(TupleInferenceModel, self)._set_padding_lengths(padding_lengths)
        # The number of tuple slots determines the shape of some of the weights in our model, so we
        # need to keep this constant.
        if self.num_tuple_slots is None:
            self.num_tuple_slots = padding_lengths['num_slots']
        if self.data_generator is not None and self.data_generator.dynamic_padding:
            return
        if self.num_question_tuples is None:
            self.num_question_tuples = padding_lengths['num_question_tuples']
        if self.num_background_tuples is None:
            self.num_background_tuples = padding_lengths['num_background_tuples']
        if self.num_slot_words is None:
            self.num_slot_words = padding_lengths['num_sentence_words']
        if self.num_options is None:
            self.num_options = padding_lengths['num_options']

    @overrides
    def get_padding_memory_scaling(self, padding_lengths: Dict[str, int]) -> int:
        num_question_tuples = padding_lengths['num_question_tuples']
        num_background_tuples = padding_lengths['num_background_tuples']
        num_sentence_words = padding_lengths['num_sentence_words']
        num_options = padding_lengths['num_options']
        return num_question_tuples * num_background_tuples * num_sentence_words * num_options

    @overrides
    def _set_padding_lengths_from_model(self):
        self.num_background_tuples = self.model.get_input_shape_at(0)[1][1]
        self.num_options = self.model.get_input_shape_at(0)[0][1]
        self.num_question_tuples = self.model.get_input_shape_at(0)[0][2]
        self.num_tuple_slots = self.model.get_input_shape_at(0)[0][3]
        self.num_slot_words = self.model.get_input_shape_at(0)[0][4]
        self._set_text_lengths_from_model_input = self.model.get_input_shape_at(0)[0][4:]

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
        choice tuple::

            :math:`s_i^c = 1 - \prod_{j=1:J}(1 - q_1 * s_{ij}^c)`

        where q_1 is the noise parameter for this first noisy-or.  Next, we combine these scores for each answer
        choice again using the noisy-or to get the entailment score for the answer candidate::

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
        tiled_question = RepeatLike(axis=3, copy_from_axis=1)([question_input, background_input])

        # Expand and tile the background input to match question input.
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples, num_tuple_slots,
        #               num_slot_words)
        # First, add num_options.
        tiled_background = RepeatLike(axis=1, copy_from_axis=1)([background_input, question_input])
        # Next, add num_question_tuples.
        tiled_background = RepeatLike(axis=2, copy_from_axis=2)([tiled_background, question_input])

        # Find the matches between the question and background tuples.
        # shape: (batch size, num_options, num_question_tuples, num_background_tuples)
        matches = self.tuple_matcher([tiled_question, tiled_background])

        # Find the probability that any given question tuple is entailed by the given background tuples.
        # shape: (batch size, num_options, num_question_tuples)
        combine_background_evidence = NoisyOr(axis=-1, param_init=self.noisy_or_param_init)
        combine_background_evidence.name = "noisy_or_1"
        qi_probabilities = combine_background_evidence(matches)

        # If desired, peek across the options, and normalize the amount that a given answer tuple template "counts"
        # towards a correct answer.
        if self.normalize_tuples_across_answers:
            normalize_across_options = SubtractMinimum(axis=1)
            qi_probabilities = normalize_across_options(qi_probabilities)

        # Find the probability that any given option is correct, given the entailement scores of each of its
        # question tuples given the set of background tuples.
        # shape: (batch size, num_options)
        combine_question_evidence = NoisyOr(axis=-1, param_init=self.noisy_or_param_init)
        combine_question_evidence.name = "noisy_or_2"
        options_probabilities = combine_question_evidence(qi_probabilities)

        # Softmax over the options to choose the best one.
        final_output = MaskedSoftmax(name="masked_softmax")(options_probabilities)

        return DeepQaModel(input=[question_input, background_input], output=[final_output])

    @overrides
    def _instance_debug_output(self, instance: TupleInferenceInstance, outputs: Dict[str, numpy.array]) -> str:
        num_to_display = 5
        result = ""
        result += "\n====================================================================\n"
        result += "Instance: %s\n" % instance.index
        result += "Question Text: %s\n" % instance.question_text
        result += "Label: %s\n" % instance.label
        result += "Num tuples per answer option: %s\n" % [len(answer) for answer in instance.answer_tuples]
        result += "(limiting display to top %s at various levels)\n" % num_to_display
        result += "====================================================================\n"

        answer_scores = []
        index_of_chosen = None
        softmax_output = outputs.get("masked_softmax", None)
        if softmax_output is not None:
            answer_scores = list(enumerate(softmax_output))
            sorted_answer_scores = sorted(answer_scores, key=lambda tup: tup[1], reverse=True)
            # TODO(becky): not handling ties
            index_of_chosen = sorted_answer_scores[0][0]

        result += "Final scores: %s\n" % answer_scores
        if index_of_chosen is None:
            result += "ERROR: no answer chosen\n"
        elif index_of_chosen == instance.label:
            result += "  Answered correctly!\n"
        else:
            result += "  Answered incorrectly\n"
        result += "====================================================================\n"

        # Output of the tuple matcher layer:
        # shape: (num_options, num_question_tuples, num_background_tuples)
        tuple_matcher_output = outputs.get('match_layer', None)
        if tuple_matcher_output is not None:
            # correct answer:
            # Keep only the first tuples (depending on model setting) because when we padded we set
            # truncate_from_right to False.
            correct_tuples = instance.answer_tuples[instance.label][:self.num_question_tuples]
            background_tuples = instance.background_tuples[:self.num_background_tuples]
            result += "-----------------------------------\n"
            result += " GOLD ANSWER: (Final score: {0})\n".format(answer_scores[instance.label][1])
            result += "-----------------------------------\n"
            result += self._render_tuple_match_scores(correct_tuples,
                                                      background_tuples,
                                                      tuple_matcher_output[instance.label],
                                                      instance)

            result += "-------------------\n"
            result += " Incorrect Answers: \n"
            result += "-------------------\n"
            # NOTE: that extra padded "options" are added on the right, so this should be fine.
            for option in range(len(instance.answer_tuples)):
                chosen_status = ""
                if option != instance.label:
                    option_tuples = instance.answer_tuples[option][:self.num_question_tuples]
                    if option == index_of_chosen:
                        chosen_status = "(Chosen)"
                    result += "\nOption {0} {1}: (Final Score: {2})\n".format(option,
                                                                              chosen_status,
                                                                              answer_scores[option][1])
                    result += self._render_tuple_match_scores(option_tuples,
                                                              background_tuples,
                                                              tuple_matcher_output[option],
                                                              instance)
        result += "\n"

        return result

    def _render_tuple_match_scores(self, answer_tuples, background_tuples, tuple_matcher_output, instance):
        result = ""
        for i, answer_tuple in enumerate(answer_tuples):
            answer_tuple_string = "\n\t".join(textwrap.wrap(answer_tuple.display_string(), self.display_text_wrap))
            result += "Question (repeated): %s\n" % instance.question_text
            result += "Answer_tuple_{0} : \n\t{1}\n\n".format(i, answer_tuple_string)
            result += "Top {0} (out of {1}) highest scoring background tuples:\n\n".format(self.display_num_tuples,
                                                                                           len(background_tuples))
            tuple_match_scores = []
            for j, background_tuple in enumerate(background_tuples):
                tuple_match_score = tuple_matcher_output[i, j]
                tuple_match_scores.append((tuple_match_score, j, background_tuple))
            # Sort descending by tuple match score
            sorted_by_score = sorted(tuple_match_scores, key=lambda tup: tup[0],
                                     reverse=True)[:self.display_num_tuples]
            for scored in sorted_by_score:
                background_tuple_index = scored[1]
                background_tuple_string = scored[2].display_string()
                wrapped_tuple = "\n\t".join(textwrap.wrap(background_tuple_string, self.display_text_wrap))
                result += "  (TupleMatch Score %s) " % scored[0]
                result += "\tbg_tuple_{0} \n\t{1}\n".format(background_tuple_index, wrapped_tuple)
            result += "\n"
        return result
