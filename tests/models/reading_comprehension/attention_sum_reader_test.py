# pylint: disable=no-self-use,invalid-name

from deep_qa.models.reading_comprehension import AttentionSumReader
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestAttentionSumReader(DeepQaTestCase):
    def test_train_does_not_crash_and_load_works(self):
        self.write_who_did_what_files()
        args = Params({
                'save_models': True,
                "encoder": {
                        "default": {
                                "type": "bi_gru",
                                "units": 7
                        }
                },
                "seq2seq_encoder": {
                        "default": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 7
                                },
                                "wrapper_params": {}
                        }
                },
                "embedding_dim": {"words": 5},
        })
        self.ensure_model_trains_and_loads(AttentionSumReader, args)
