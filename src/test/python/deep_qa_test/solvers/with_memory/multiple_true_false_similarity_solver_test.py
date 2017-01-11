# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.with_memory.multiple_true_false_similarity import MultipleTrueFalseSimilaritySolver
from ...common.constants import TEST_DIR
from ...common.solvers import get_solver
from ...common.solvers import write_multiple_true_false_memory_network_files


class TestMultipleTrueFalseSimilaritySolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_multiple_true_false_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MultipleTrueFalseSimilaritySolver)
        solver.train()
