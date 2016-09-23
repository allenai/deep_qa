# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from dlfa.solvers.pretraining.snli_pretrainer import SnliEntailmentPretrainer
from dlfa.solvers.pretraining.snli_pretrainer import SnliAttentionPretrainer
from dlfa.solvers.memory_network import MemoryNetworkSolver
from dlfa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from ...common.constants import TEST_DIR
from ...common.constants import SNLI_FILE
from ...common.solvers import get_solver
from ...common.solvers import write_memory_network_files
from ...common.solvers import write_snli_file


class TestSnliPretrainers(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.mkdir(TEST_DIR)
        write_snli_file()
        write_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_entailment_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, SNLI_FILE)
        pretrainer.train()

    def test_entailment_pretraining_does_not_crash_with_multiple_choice_memory_network_solver(self):
        solver = get_solver(MultipleChoiceMemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, SNLI_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, SNLI_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_multiple_choice_memory_network_solver(self):
        solver = get_solver(MultipleChoiceMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, SNLI_FILE)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_question_answer_memory_network_solver(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, SNLI_FILE)
        pretrainer.train()

    def test_solver_training_works_after_pretraining(self):
        # TODO(matt): It's possible in this test that the pretrainers don't actually get called,
        # and we wouldn't know it.  Not sure how to make sure that the pretrainers are actually
        # called in this test.  You could probably do it with some complicated use of patching...
        args = {
                'snli_file': SNLI_FILE,
                'pretrain_entailment': None,
                'pretrain_attention': None,
                }
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()
