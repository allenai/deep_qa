# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.multiple_choice_similarity import MultipleChoiceSimilaritySolver
from ..common.constants import TEST_DIR
from ..common.solvers import get_solver
from ..common.solvers import write_multiple_choice_memory_network_files


class TestMultipleChoiceSimilaritySolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_multiple_choice_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MultipleChoiceSimilaritySolver)
        solver.train()
