# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.models.multiple_choice_qa.multiple_true_false_similarity import MultipleTrueFalseSimilarity
from ...common.constants import TEST_DIR
from ...common.models import get_model
from ...common.models import write_multiple_true_false_memory_network_files
from ...common.test_markers import requires_tensorflow


class TestMultipleTrueFalseSimilarity(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_multiple_true_false_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    @requires_tensorflow
    def test_train_does_not_crash(self):
        model = get_model(MultipleTrueFalseSimilarity)
        model.train()
