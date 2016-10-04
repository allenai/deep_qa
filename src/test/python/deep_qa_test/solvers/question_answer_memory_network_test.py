# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from ..common.constants import TEST_DIR
from ..common.solvers import get_solver
from ..common.solvers import write_question_answer_memory_network_files


class TestQuestionAnswerMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_question_answer_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': {'type': 'parameterized'}}
        solver = get_solver(QuestionAnswerMemoryNetworkSolver, args)
        solver.train()
