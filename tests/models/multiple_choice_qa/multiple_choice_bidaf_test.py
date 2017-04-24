# pylint: disable=no-self-use,invalid-name
from keras import backend as K
import pytest

from deep_qa.contrib.models.multiple_choice_bidaf import MultipleChoiceBidaf
from deep_qa.models.reading_comprehension import BidirectionalAttentionFlow
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestMultipleChoiceBidaf(DeepQaTestCase):
    @pytest.mark.skip(reason="There is some Keras 2 bug that blocks this model from working.  "
                      "Checkout an Keras 1 version of the code if you want to use this model, or "
                      "figure out and fix the Keras 2 bug (or wait for someone else to do it...).")
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = Params({
                'model_serialization_prefix': self.TEST_DIR + "_bidaf",
                'embedding_dim': {"words": 4, "characters": 4},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                })
        bidaf_model = self.get_model(BidirectionalAttentionFlow, args)
        bidaf_model.train()
        K.clear_session()

        bidaf_model_params = self.get_model_params(BidirectionalAttentionFlow, args)
        args = Params({
                'bidaf_params': bidaf_model_params,
                'train_bidaf': False,
                'similarity_function': {'type': 'linear', 'combination': 'x,y'},
                })
        self.write_who_did_what_files()
        model, _ = self.ensure_model_trains_and_loads(MultipleChoiceBidaf, args)
        # All of the params come from the linear similarity function in the attention layer,
        # because we set `train_bidaf` to `False`.  41 comes from 32 + 8 + 1, where 32 is from the
        # modeled passage (see the equations in the paper for why it's 32), 8 is from the Bi-LSTM
        # operating on the answer options (embedding_dim * 2), and 1 is from the bias.
        assert sum([K.count_params(p) for p in model.model.trainable_weights]) == 41
