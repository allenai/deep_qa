# pylint: disable=no-self-use,invalid-name
import argparse
import codecs

from pyfakefs import fake_filesystem_unittest
import pytest

import numpy

from dlfa.data.dataset import TextDataset
from dlfa.data.instance import TextInstance
from dlfa.solvers.lstm_solver import LSTMSolver


class TestNNSolver(fake_filesystem_unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.setUpPyfakefs()
        self.validation_file = '/validation_file'
        with codecs.open(self.validation_file, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tq1a1\t0\n')
            validation_file.write('2\tq1a2\t1\n')
            validation_file.write('3\tq1a3\t0\n')
            validation_file.write('4\tq1a4\t0\n')
            validation_file.write('5\tq2a1\t0\n')
            validation_file.write('6\tq2a2\t0\n')
            validation_file.write('7\tq2a3\t1\n')
            validation_file.write('8\tq2a4\t0\n')
            validation_file.write('9\tq3a1\t0\n')
            validation_file.write('10\tq3a2\t0\n')
            validation_file.write('11\tq3a3\t0\n')
            validation_file.write('12\tq3a4\t1\n')
        self.train_file = '/train_file'
        with codecs.open(self.train_file, 'w', 'utf-8') as train_file:
            train_file.write('1\tsentence1\t0\n')
            train_file.write('2\tsentence2\t1\n')
            train_file.write('3\tsentence3\t0\n')
            train_file.write('4\tsentence4\t1\n')
            train_file.write('5\tsentence5\t0\n')
            train_file.write('6\tsentence6\t0\n')

    def _get_solver(self):
        # We'll use the LSTMSolver for these tests, so we have a class that actually has fully
        # implemented methods.  We use the argument parser because it's easiest to get default
        # values for all of the parameters this way.
        parser = argparse.ArgumentParser()
        LSTMSolver.update_arg_parser(parser)
        additional_arguments = [
                '--train_file', self.train_file,
                '--validation_file', self.validation_file,
                '--model_serialization_prefix', '/',
                ]
        args = parser.parse_args(additional_arguments)
        return LSTMSolver(**vars(args))

    def test_prep_question_data_does_not_shuffle_data(self):
        dataset = TextDataset.read_from_file(self.validation_file)
        solver = self._get_solver()
        _, labels = solver._prep_question_dataset(dataset)
        assert numpy.all(labels == numpy.asarray([1, 2, 3]))

    def test_prep_question_data_crashes_on_invalid_question_data_not_divisible_by_four(self):
        solver = self._get_solver()
        dataset = TextDataset([])
        dataset.instances.append(TextInstance("instance1", label=True))
        with pytest.raises(Exception):
            solver._prep_question_dataset(dataset)

    def test_prep_question_data_crashes_on_invalid_question_data_more_than_one_correct_answer(self):
        solver = self._get_solver()
        dataset = TextDataset([])
        dataset.instances.append(TextInstance("instance1", label=True))
        dataset.instances.append(TextInstance("instance2", label=True))
        dataset.instances.append(TextInstance("instance3", label=False))
        dataset.instances.append(TextInstance("instance4", label=False))
        with pytest.raises(Exception):
            solver._prep_question_dataset(dataset)

    def test_validation_data_loads_correctly(self):
        solver = self._get_solver()
        _, labels = solver._get_validation_data()
        assert numpy.all(labels == numpy.asarray([1, 2, 3]))
