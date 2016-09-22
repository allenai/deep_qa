# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import codecs
import os
import shutil

from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from ..common.constants import TEST_DIR
from ..common.constants import TRAIN_FILE
from ..common.constants import TRAIN_BACKGROUND
from ..common.constants import VALIDATION_FILE
from ..common.constants import VALIDATION_BACKGROUND
from ..common.solvers import get_solver


class TestQuestionAnswerMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\tquestion1\tanswer1###answer2\t0\n')
        with codecs.open(VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
            validation_background.write('1\tvb1\tvb2\n')
        with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\ta b e i d\tanswer1###answer2\t0\n')
            train_file.write('2\ta b c d\tanswer3###answer4\t1\n')
            train_file.write('3\te d w f d s a\tanswer5###answer6###answer9\t2\n')
            train_file.write('4\te fj k w q\tanswer7###answer8\t0\n')
        with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
            train_background.write('1\tsb1\tsb2\n')
            train_background.write('2\tsb3\n')
            train_background.write('3\tsb4\n')
            train_background.write('4\tsb5\tsb6\n')

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': 'parameterized'}
        solver = get_solver(QuestionAnswerMemoryNetworkSolver, args)
        solver.train()
