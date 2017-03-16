# pylint: disable=no-self-use,invalid-name
from keras import backend as K

from deep_qa.models.multiple_choice_qa.multiple_choice_bidaf import MultipleChoiceBidaf
from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...common.test_case import DeepQaTestCase
from ...common.test_markers import requires_tensorflow


class TestMultipleChoiceBidaf(DeepQaTestCase):
    # Theano has some problems with saving the loaded model in h5, because of how it names weights
    # (pretty sure this arises because we're using two different BiDAF-based submodels in our
    # model).  We'll punt on fixing that for now; the fix would likely be a pretty major hack.
    # Theano can _train_ the model alright, but it can't _save_ it, or (presumably) load it.
    @requires_tensorflow
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = {
                'model_serialization_prefix': self.TEST_DIR + "_bidaf",
                'embedding_dim': {"words": 4, "characters": 4},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                }
        bidaf_model = self.get_model(BidirectionalAttentionFlow, args)
        bidaf_model.train()

        bidaf_model_params = self.get_model_params(BidirectionalAttentionFlow, args)
        args = {
                'bidaf_params': bidaf_model_params,
                'train_bidaf': False,
                'similarity_function': {'type': 'linear', 'combination': 'x,y'},
                }
        self.write_who_did_what_files()
        model, _ = self.ensure_model_trains_and_loads(MultipleChoiceBidaf, args)
        # All of the params come from the linear similarity function in the attention layer,
        # because we set `train_bidaf` to `False`.  41 comes from 32 + 8 + 1, where 32 is from the
        # modeled passage (see the equations in the paper for why it's 32), 8 is from the Bi-LSTM
        # operating on the answer options (embedding_dim * 2), and 1 is from the bias.
        assert sum([K.count_params(p) for p in model.model.trainable_weights]) == 41
