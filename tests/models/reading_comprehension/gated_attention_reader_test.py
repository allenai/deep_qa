# pylint: disable=no-self-use,invalid-name

from deep_qa.models.reading_comprehension import GatedAttentionReader
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestGatedAttention(DeepQaTestCase):
    def test_cloze_train_does_not_crash(self):
        self.write_who_did_what_files()
        args = Params({
                'save_models': True,
                "qd_common_feature": True,
                "gating_function": "+",
                "cloze_token": "xxxxx",
                "num_gated_attention_layers": 2,
                "tokenizer": {
                        "type": "words and characters"
                },
                "encoder": {
                        "word": {
                                "type": "bi_gru",
                                "units": 2,
                        }
                },
                "seq2seq_encoder": {
                        "question_0": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        },
                        "document_0": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        },
                        "document_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        },
                        "question_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {
                                        "merge_mode": None
                                }
                        }
                },
                "embedding_dim": {"words": 4, "characters": 4},
        })
        model, loaded_model = self.ensure_model_trains_and_loads(GatedAttentionReader, args)
        # verify that the gated attention function was set properly
        assert model.gating_function == "+"
        assert model.gating_function == model.model.get_layer("gated_attention_0").gating_function

        # verify that the gated attention function was set properly in the loaded model
        assert loaded_model.gating_function == "+"
        assert loaded_model.gating_function == loaded_model.model.get_layer("gated_attention_0").gating_function

    def test_non_cloze_train_does_not_crash(self):
        self.write_who_did_what_files()
        args = Params({
                'save_models': True,
                "qd_common_feature": True,
                "num_gated_attention_layers": 2,
                "gating_function": "+",
                "tokenizer": {
                        "type": "words and characters"
                },
                "encoder": {
                        "word": {
                                "type": "bi_gru",
                                "units": 2,
                        },
                        "question_final": {
                                "type": "bi_gru",
                                "units": 3
                        }

                },
                "seq2seq_encoder": {
                        "question_0": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        },
                        "document_0": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        },
                        "document_final": {
                                "type": "bi_gru",
                                "encoder_params": {
                                        "units": 3
                                },
                                "wrapper_params": {}
                        }
                },
                "embedding_dim": {"words": 4, "characters": 4},
        })
        model, loaded_model = self.ensure_model_trains_and_loads(GatedAttentionReader, args)
        # verify that the gated attention function was set properly
        assert model.gating_function == "+"
        assert model.gating_function == model.model.get_layer("gated_attention_0").gating_function

        # verify that the gated attention function was set properly in the loaded model
        assert loaded_model.gating_function == "+"
        assert loaded_model.gating_function == loaded_model.model.get_layer("gated_attention_0").gating_function
