# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import logging
import os
import shutil

from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from deep_qa.common.checks import log_keras_version_info
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_span_prediction_files
from ...common.test_markers import requires_tensorflow


class TestBidirectionalAttentionFlow(TestCase):
    def setUp(self):
        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                            level=logging.INFO)
        log_keras_version_info()
        os.makedirs(TEST_DIR, exist_ok=True)
        write_span_prediction_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    @requires_tensorflow
    def test_train_does_not_crash(self):
        args = {
                'show_summary_with_masking_info': True,
                }
        solver = get_model(BidirectionalAttentionFlow, args)
        solver.train()
