# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_allclose

from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...common.test_case import DeepQaTestCase


class TestBidirectionalAttentionFlow(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = {
                'save_models': True,
                'show_summary_with_masking_info': True,
                }
        model = self.get_model(BidirectionalAttentionFlow, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(BidirectionalAttentionFlow, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))
