# pylint: disable=no-self-use,invalid-name

from unittest.mock import patch
import os

from pyfakefs import fake_filesystem_unittest
import pytest

import numpy

from deep_qa.data.dataset import TextDataset
from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.text_instance import TrueFalseInstance
from deep_qa.data.tokenizer import NltkTokenizer, SimpleTokenizer
from deep_qa.solvers.lstm_solver import LSTMSolver
from ..common.constants import TEST_DIR
from ..common.constants import VALIDATION_FILE
from ..common.solvers import get_solver
from ..common.solvers import write_lstm_solver_files


class TestNNSolver(fake_filesystem_unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.setUpPyfakefs()
        os.mkdir(TEST_DIR)
        write_lstm_solver_files()

    @patch.object(DataIndexer, 'fit_word_dictionary', lambda x, y: y)
    @patch.object(LSTMSolver, 'prep_labeled_data', return_value=([], numpy.asarray([[1, 0], [0, 1],
                                                                                    [0, 1], [0, 1]])))
    def test_tokenizer_argument_is_handled(self, prep_method):  # pylint: disable=unused-argument
        solver = get_solver(LSTMSolver, {'tokenizer': 'nltk'})
        assert isinstance(solver.tokenizer, NltkTokenizer)
        solver._get_training_data()
        assert isinstance(solver.training_dataset.instances[0].tokenizer, NltkTokenizer)
        solver._get_validation_data()
        assert isinstance(solver.validation_dataset.instances[0].tokenizer, NltkTokenizer)
        solver = get_solver(LSTMSolver, {'tokenizer': 'simple'})
        assert isinstance(solver.tokenizer, SimpleTokenizer)
        solver._get_training_data()
        assert isinstance(solver.training_dataset.instances[0].tokenizer, SimpleTokenizer)
        solver._get_validation_data()
        assert isinstance(solver.validation_dataset.instances[0].tokenizer, SimpleTokenizer)

    def test_prep_question_data_does_not_shuffle_data(self):
        dataset = TextDataset.read_from_file(VALIDATION_FILE, TrueFalseInstance)
        solver = get_solver(LSTMSolver)
        _, labels = solver._prep_question_dataset(dataset)
        assert numpy.all(labels == numpy.asarray([1, 2, 3]))

    def test_prep_question_data_crashes_on_invalid_question_data_not_divisible_by_four(self):
        solver = get_solver(LSTMSolver)
        dataset = TextDataset([])
        dataset.instances.append(TrueFalseInstance("instance1", label=True))
        with pytest.raises(Exception):
            solver._prep_question_dataset(dataset)

    def test_prep_question_data_crashes_on_invalid_question_data_more_than_one_correct_answer(self):
        solver = get_solver(LSTMSolver)
        dataset = TextDataset([])
        dataset.instances.append(TrueFalseInstance("instance1", label=True))
        dataset.instances.append(TrueFalseInstance("instance2", label=True))
        dataset.instances.append(TrueFalseInstance("instance3", label=False))
        dataset.instances.append(TrueFalseInstance("instance4", label=False))
        with pytest.raises(Exception):
            solver._prep_question_dataset(dataset)

    def test_validation_data_loads_correctly(self):
        solver = get_solver(LSTMSolver)
        _, labels = solver._get_validation_data()
        assert numpy.all(labels == numpy.asarray([1, 2, 3]))
