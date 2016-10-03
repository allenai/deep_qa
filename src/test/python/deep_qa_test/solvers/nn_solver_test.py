# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.data.tokenizer import NltkTokenizer, SimpleTokenizer
from deep_qa.solvers.lstm_solver import LSTMSolver
from ..common.constants import TEST_DIR
from ..common.constants import TRAIN_FILE
from ..common.solvers import get_solver
from ..common.solvers import write_lstm_solver_files


class TestNNSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_lstm_solver_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_tokenizer_argument_is_handled(self):  # pylint: disable=unused-argument
        solver = get_solver(LSTMSolver, {'tokenizer': 'nltk'})
        assert isinstance(solver.tokenizer, NltkTokenizer)
        dataset = solver._load_dataset_from_files([TRAIN_FILE])
        assert isinstance(dataset.instances[0].tokenizer, NltkTokenizer)

        solver = get_solver(LSTMSolver, {'tokenizer': 'simple'})
        assert isinstance(solver.tokenizer, SimpleTokenizer)
        dataset = solver._load_dataset_from_files([TRAIN_FILE])
        assert isinstance(dataset.instances[0].tokenizer, SimpleTokenizer)
