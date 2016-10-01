# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from dlfa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from ..common.constants import TEST_DIR
from ..common.solvers import get_solver
from ..common.solvers import write_multiple_choice_memory_network_files


class TestMultipleChoiceMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_multiple_choice_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MultipleChoiceMemoryNetworkSolver)
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': 'parameterized'}
        solver = get_solver(MultipleChoiceMemoryNetworkSolver, args)
        solver.train()
