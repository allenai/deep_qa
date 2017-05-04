from collections import OrderedDict
from typing import Any, Dict

from ...common.params import pop_choice
from ..backend import CollapseToBatch, ExpandFromBatch, Squeeze
from .word_overlap_tuple_matcher import WordOverlapTupleMatcher


class WordTokenTupleMatcher:
    """
    ``WordTokenTupleMatchers`` operate on tuples with shape ``(batch_size, num_slots, num_words)``.
    This class gets two inputs of shape ``(batch_size, num_options, num_question_tuples,
    num_background_tuples, num_slots, num_words)``, collapses them down to the right shape using
    ``TimeDistributed`` (or equivalent) and passes them through the actual tuple matcher, returning
    a tensor of tuple matches of shape ``(batch_size, num_options, num_question_tuples,
    num_background_tuples)``.

    Parameters
    ----------
    text_trainer: TextTrainer
        A reference to the ``TextTrainer`` object this ``TupleMatcher`` belongs to.  We don't
        actually use this here, but we take the parameter for API consistency.

    tuple_matcher_params: Dict[str, Any], default={}
        Parameters for constructing an underlying ``TupleMatcher`` that operates on word token
        tuples.  We only read the "type" key here, which indexes a class in
        ``word_token_tuple_matchers``, and then pass the rest of the parameters on to that class as
        ``kwargs``.
    """
    def __init__(self, text_trainer, tuple_matcher_params: Dict[str, Any]=None):
        self.text_trainer = text_trainer  # currently unused
        if tuple_matcher_params is None:
            tuple_matcher_params = {}
        tuple_matcher_choice = pop_choice(tuple_matcher_params, "matcher_type",
                                          list(word_token_tuple_matchers.keys()),
                                          default_to_first_choice=True)
        self.tuple_matcher = word_token_tuple_matchers[tuple_matcher_choice](**tuple_matcher_params)

    def __call__(self, inputs):
        tuple1, tuple2 = inputs
        collapsed_tuple1 = CollapseToBatch(num_to_collapse=3)(tuple1)
        collapsed_tuple2 = CollapseToBatch(num_to_collapse=3)(tuple2)
        collapsed_matches = self.tuple_matcher([collapsed_tuple1, collapsed_tuple2])
        matches = ExpandFromBatch(num_to_expand=3)([collapsed_matches, tuple1])
        matches = Squeeze()(matches)
        return matches

    @staticmethod
    def get_custom_objects():
        return {
                'CollapseToBatch': CollapseToBatch,
                'ExpandFromBatch': ExpandFromBatch,
                'Squeeze': Squeeze,
                'WordOverlapTupleMatcher': WordOverlapTupleMatcher,
                }



# The first item added here will be used as the default in some cases.
word_token_tuple_matchers = OrderedDict()  # pylint: disable=invalid-name
word_token_tuple_matchers['word_overlap'] = WordOverlapTupleMatcher
