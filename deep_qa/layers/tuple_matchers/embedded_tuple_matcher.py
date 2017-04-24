from collections import OrderedDict
from typing import Any, Dict

from ...common.params import pop_choice
from ..wrappers.time_distributed_with_mask import TimeDistributedWithMask
from .threshold_tuple_matcher import ThresholdTupleMatcher


class EmbeddedTupleMatcher:
    """
    ``EmbeddedTupleMatchers`` operate on tuples with shape ``(batch_size, num_slots, num_words,
    embedding_dim)``.  This class embeds the tuple input, then passes it off to an underlying
    ``Layer`` that computes the match between two tuples with the above shape.

    Parameters
    ----------
    text_trainer: TextTrainer
        A reference to the TextTrainer object this TupleMatcher belongs to, so we can call
        ``embed_input`` with it.

    tuple_matcher_params: Dict[str, Any], default={}
        Parameters for constructing an underlying TupleMatcher that operates on embedded tuples.
        We only read the "type" key here, which indexes a class in ``embedded_tuple_matchers``, and
        then pass the rest of the parameters on to that class as ``kwargs``.
    """
    def __init__(self, text_trainer, tuple_matcher_params: Dict[str, Any]=None):
        self.text_trainer = text_trainer
        if tuple_matcher_params is None:
            tuple_matcher_params = {}
        tuple_matcher_choice = pop_choice(tuple_matcher_params, "embedded_matcher_type",
                                          list(embedded_tuple_matchers.keys()),
                                          default_to_first_choice=True)
        self.tuple_matcher = embedded_tuple_matchers[tuple_matcher_choice](**tuple_matcher_params)

    def __call__(self, inputs):
        # pylint: disable=protected-access
        tuple1, tuple2 = inputs
        embedded_tuple1 = self.text_trainer._embed_input(tuple1)
        embedded_tuple2 = self.text_trainer._embed_input(tuple2)
        # The three TimeDistributedWithMasks wrap around the first three dimensions of the inputs:
        # num_options, num_answer_tuple, and num_background_tuples.
        match_layer = TimeDistributedWithMask(TimeDistributedWithMask(TimeDistributedWithMask(self.tuple_matcher)))
        return match_layer([embedded_tuple1, embedded_tuple2])


# The first item added here will be used as the default in some cases.
embedded_tuple_matchers = OrderedDict()  # pylint: disable=invalid-name
embedded_tuple_matchers['threshold'] = ThresholdTupleMatcher
