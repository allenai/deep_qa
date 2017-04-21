# pylint: disable=no-self-use,invalid-name
import numpy
from flaky import flaky

from deep_qa.models.reading_comprehension import BidirectionalAttentionFlow
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestBidirectionalAttentionFlow(DeepQaTestCase):
    @flaky
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = Params({
                'embedding_dim': {'words': 4, 'characters': 4},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                })
        self.ensure_model_trains_and_loads(BidirectionalAttentionFlow, args)

    def test_get_best_span(self):
        # Note that the best span cannot be (1, 0) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = numpy.array([0.1, 0.3, 0.05, 0.3, 0.25])
        span_end_probs = numpy.array([0.5, 0.1, 0.2, 0.05, 0.15])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (1, 2)

        # Testing an edge case of the dynamic program here, for the order of when you update the
        # best previous span position.  We should not get (1, 1), because that's an empty span.
        span_begin_probs = numpy.array([0.4, 0.5, 0.1])
        span_end_probs = numpy.array([0.3, 0.6, 0.1])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (0, 1)

        # test higher-order input
        # Note that the best span cannot be (1, 1) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = numpy.array([[0.1, 0.3, 0.05, 0.3, 0.25]])
        span_end_probs = numpy.array([[0.1, 0.5, 0.2, 0.05, 0.15]])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (1, 2)
