# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.models.multiple_choice_qa.question_answer_memory_network import QuestionAnswerMemoryNetwork
from ...common.constants import TEST_DIR
from ...common.models import get_model
from ...common.models import write_question_answer_memory_network_files


class TestQuestionAnswerMemoryNetwork(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_question_answer_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        model = get_model(QuestionAnswerMemoryNetwork)
        model.train()
