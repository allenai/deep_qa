# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_allclose

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
                "embedding_size": 5,
        }
        model = self.get_model(AttentionSumReader, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(AttentionSumReader, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))
