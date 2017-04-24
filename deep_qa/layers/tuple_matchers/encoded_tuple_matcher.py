from collections import OrderedDict
from typing import Any, Dict

from ...layers.wrappers.encoder_wrapper import EncoderWrapper
from ...layers.wrappers.time_distributed_with_mask import TimeDistributedWithMask
from ...common.params import pop_choice
from .slot_similarity_tuple_matcher import SlotSimilarityTupleMatcher


class EncodedTupleMatcher:
    """
    ``EncodedTupleMatchers`` operate on tuples with shape ``(batch_size, num_slots, embed dimension)``.
    In order to do so, this class encodes the original tuple input (using a ``TextTrainer`` object),
    then passes it off to an underlying ``Layer`` that computes the match between two tuples with the
    above shape.

    Parameters
    ----------
    text_trainer: TextTrainer
        A reference to the TextTrainer object this TupleMatcher belongs to, so we can call
        ``embed_input`` and ``get_encoder`` with it.

    tuple_matcher_params: Dict[str, Any], default={}
        Parameters for constructing an underlying TupleMatcher that operates on encoded tuples.
        We only read the "type" key here, which indexes a class in ``encoded_tuple_matchers``, and
        then pass the rest of the parameters on to that class as ``kwargs``.
    """
    def __init__(self, text_trainer, tuple_matcher_params: Dict[str, Any]=None):
        self.text_trainer = text_trainer
        if tuple_matcher_params is None:
            tuple_matcher_params = {}
        tuple_matcher_choice = pop_choice(tuple_matcher_params, "encoded_matcher_type",
                                          list(encoded_tuple_matchers.keys()),
                                          default_to_first_choice=True)
        self.tuple_matcher = encoded_tuple_matchers[tuple_matcher_choice](**tuple_matcher_params)

    def __call__(self, inputs):
        # pylint: disable=protected-access
        tuple1, tuple2 = inputs
        embedded_tuple1 = self.text_trainer._embed_input(tuple1)
        embedded_tuple2 = self.text_trainer._embed_input(tuple2)
        tuple_encoder = self.text_trainer._get_encoder(name="tuples",
                                                       fallback_behavior="use default encoder")
        # We use separate encoders here in case the tuples have different shapes (e.g., different
        # numbers of slots, or number of words per slot).
        tuple1_encoder = EncoderWrapper(EncoderWrapper(EncoderWrapper(EncoderWrapper(tuple_encoder),
                                                                      name="tuple1_encoder")))
        tuple2_encoder = EncoderWrapper(EncoderWrapper(EncoderWrapper(EncoderWrapper(tuple_encoder),
                                                                      name="tuple2_encoder")))
        encoded_tuple1 = tuple1_encoder(embedded_tuple1)
        encoded_tuple2 = tuple2_encoder(embedded_tuple2)
        # The three TimeDistributedWithMask wrap around the first three dimensions of the inputs:
        # num_options, num_answer_tuple, and num_background_tuples.
        match_layer = TimeDistributedWithMask(TimeDistributedWithMask(TimeDistributedWithMask(self.tuple_matcher)))
        return match_layer([encoded_tuple1, encoded_tuple2])


# The first item added here will be used as the default in some cases.
encoded_tuple_matchers = OrderedDict()  # pylint: disable=invalid-name
encoded_tuple_matchers['slot_similarity'] = SlotSimilarityTupleMatcher
