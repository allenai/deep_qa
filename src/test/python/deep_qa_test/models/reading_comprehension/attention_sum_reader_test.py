# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import os
import logging
import shutil
from numpy.testing import assert_allclose

from deep_qa.models.reading_comprehension.attention_sum_reader import AttentionSumReader
from deep_qa.common.checks import log_keras_version_info
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_who_did_what_files


class TestAttentionSumReader(TestCase):
    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.INFO)
        log_keras_version_info()
        os.makedirs(TEST_DIR, exist_ok=True)
        write_who_did_what_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash_and_load_works(self):
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
        model = get_model(AttentionSumReader, args)
        model.train()

        # load the model that we serialized
        loaded_model = get_model(AttentionSumReader, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))
