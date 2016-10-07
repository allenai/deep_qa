# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.memory_network import MemoryNetworkSolver
from ..common.constants import TEST_DIR
from ..common.solvers import get_solver
from ..common.solvers import write_memory_network_files


class TestMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MemoryNetworkSolver)
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': {'type': 'parameterized'}}
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()

    def test_train_does_not_crash_with_cnn_encoder(self):
        args = {'encoder': {'type': 'cnn', 'ngram_filter_sizes': [1], 'num_filters': 1}}
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()

    def test_train_does_not_crash_with_positional_encoder(self):
        args = {'encoder': {'type': 'positional'}}
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()
