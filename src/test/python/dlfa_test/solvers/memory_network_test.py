# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import codecs
import os
import shutil

from dlfa.solvers.memory_network import MemoryNetworkSolver
from ..common.constants import TEST_DIR
from ..common.constants import TRAIN_FILE
from ..common.constants import TRAIN_BACKGROUND
from ..common.constants import VALIDATION_FILE
from ..common.constants import VALIDATION_BACKGROUND
from ..common.solvers import get_solver


class TestMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tq1a1\t0\n')
            validation_file.write('2\tq1a2\t1\n')
            validation_file.write('3\tq1a3\t0\n')
            validation_file.write('4\tq1a4\t0\n')
        with codecs.open(VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
            validation_background.write('1\tvb1\tvb2\n')
            validation_background.write('2\tvb3\tvb4\tvb5\n')
            validation_background.write('3\tvb6\n')
            validation_background.write('4\tvb7\tvb8\tvb9\n')
        with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tsentence1\t0\n')
            train_file.write('2\tsentence2\t1\n')
            train_file.write('3\tsentence3\t0\n')
            train_file.write('4\tsentence4\t1\n')
            train_file.write('5\tsentence5\t0\n')
            train_file.write('6\tsentence6\t0\n')
        with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
            train_background.write('1\tsb1\tsb2\n')
            train_background.write('2\tsb3\n')
            train_background.write('3\tsb4\n')
            train_background.write('4\tsb5\tsb6\n')
            train_background.write('5\tsb7\tsb8\n')
            train_background.write('6\tsb9\n')

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MemoryNetworkSolver)
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': 'parameterized'}
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()

    def test_train_does_not_crash_with_cnn_encoder(self):
        args = {'encoder': 'cnn', 'cnn_ngram_filter_sizes': '1', 'cnn_num_filters': '1'}
        solver = self._get_solver(args)
        solver.train()
