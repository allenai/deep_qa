# pylint: disable=no-self-use,invalid-name

from deep_qa.models.sentence_selection import SiameseSentenceSelector
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestSiameseSentenceSelector(DeepQaTestCase):
    def test_model_trains_and_loads(self):
        self.write_sentence_selection_files()
        args = Params({
                'save_models': True,
                "encoder": {
                        "default": {
                                "type": "gru",
                                "units": 7
                        }
                },
                "seq2seq_encoder": {
                        "default": {
                                "type": "gru",
                                "encoder_params": {
                                        "units": 7
                                },
                                "wrapper_params": {}
                        }
                },
                "embedding_dim": {"words": 5},
        })
        self.ensure_model_trains_and_loads(SiameseSentenceSelector, args)

    def test_model_trains_and_loads_with_ranking_loss(self):
        self.write_sentence_selection_files()
        args = Params({
                'save_models': True,
                "encoder": {
                        "default": {
                                "type": "gru",
                                "units": 7
                        }
                },
                "seq2seq_encoder": {
                        "default": {
                                "type": "gru",
                                "encoder_params": {
                                        "units": 7
                                },
                                "wrapper_params": {}
                        }
                },
                "loss_function": "hinge_ranking",
                "embedding_dim": {"words": 5},
        })
        self.ensure_model_trains_and_loads(SiameseSentenceSelector, args)
