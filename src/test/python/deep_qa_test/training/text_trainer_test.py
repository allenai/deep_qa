# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.data.tokenizer import NltkTokenizer, SimpleTokenizer
from deep_qa.solvers.no_memory.true_false_solver import TrueFalseSolver
from ..common.constants import TEST_DIR
from ..common.constants import TRAIN_FILE
from ..common.solvers import get_solver
from ..common.solvers import write_true_false_solver_files
from ..common.test_markers import requires_tensorflow


class TestNNSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_true_false_solver_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_tokenizer_argument_is_handled(self):  # pylint: disable=unused-argument
        solver = get_solver(TrueFalseSolver, {'tokenizer': 'nltk'})
        assert isinstance(solver.tokenizer, NltkTokenizer)
        dataset = solver._load_dataset_from_files([TRAIN_FILE])
        assert isinstance(dataset.instances[0].tokenizer, NltkTokenizer)

        solver = get_solver(TrueFalseSolver, {'tokenizer': 'simple'})
        assert isinstance(solver.tokenizer, SimpleTokenizer)
        dataset = solver._load_dataset_from_files([TRAIN_FILE])
        assert isinstance(dataset.instances[0].tokenizer, SimpleTokenizer)

    @requires_tensorflow
    def test_tensorboard_logs_does_not_crash(self):
        solver = get_solver(TrueFalseSolver, {'tensorboard_log': TEST_DIR})
        solver.train()
