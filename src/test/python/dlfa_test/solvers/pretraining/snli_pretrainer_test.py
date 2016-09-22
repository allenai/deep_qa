# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import codecs
import os
import shutil

from dlfa.solvers.pretraining.snli_pretrainer import SnliEntailmentPretrainer
from dlfa.solvers.pretraining.snli_pretrainer import SnliAttentionPretrainer
from dlfa.solvers.memory_network import MemoryNetworkSolver
from dlfa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from ...common.constants import TEST_DIR
from ...common.constants import TRAIN_FILE
from ...common.solvers import get_solver


class TestSnliPretrainers(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
            train_file.write('1\ttext1\thypothesis1\tentails\n')
            train_file.write('2\ttext2\thypothesis2\tcontradicts\n')
            train_file.write('3\ttext3\thypothesis3\tentails\n')
            train_file.write('4\ttext4\thypothesis4\tneutral\n')
            train_file.write('5\ttext5\thypothesis5\tentails\n')
            train_file.write('6\ttext6\thypothesis6\tcontradicts\n')

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_entailment_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, TRAIN_FILE)
        pretrainer.train()

    def test_entailment_pretraining_does_not_crash_with_multiple_choice_memory_network_solver(self):
        solver = get_solver(MultipleChoiceMemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, TRAIN_FILE)
        pretrainer.train()

    def test_entailment_pretraining_does_not_crash_with_question_answer_memory_network_solver(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, TRAIN_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, TRAIN_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_multiple_choice_memory_network_solver(self):
        solver = get_solver(MultipleChoiceMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, TRAIN_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_question_answer_memory_network_solver(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, TRAIN_FILE)
        pretrainer.train()
