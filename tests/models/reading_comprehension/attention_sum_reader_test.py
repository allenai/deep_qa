# pylint: disable=no-self-use,invalid-name
from deep_qa.models.reading_comprehension.attention_sum_reader import AttentionSumReader
from ...common.test_case import DeepQaTestCase


class TestAttentionSumReader(DeepQaTestCase):
    def test_train_does_not_crash_and_load_works(self):
        self.write_who_did_what_files()
        args = {
                'save_models': True,
                "encoder": {
                        "default": {
                                "type": "bi_gru",
                                "output_dim": 7
                        }
                },
                "seq2seq_encoder": {
                        "default": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "output_dim": 7
                                },
                                "wrapper_params": {}
                        }
                },
                "embedding_dim": {"words": 5},
        }
        self.ensure_model_trains_and_loads(AttentionSumReader, args)
