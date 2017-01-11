# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.solvers.pretraining.snli_pretrainer import SnliEntailmentPretrainer
from deep_qa.solvers.pretraining.snli_pretrainer import SnliAttentionPretrainer
from deep_qa.solvers.with_memory.memory_network import MemoryNetworkSolver
from deep_qa.solvers.with_memory.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetworkSolver
from deep_qa.solvers.with_memory.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from ...common.constants import TEST_DIR
from ...common.constants import SNLI_FILE
from ...common.solvers import get_solver
from ...common.solvers import write_memory_network_files
from ...common.solvers import write_snli_file


class TestSnliPretrainers(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_snli_file()
        write_memory_network_files()
        self.pretrainer_params = {"train_files": [SNLI_FILE]}

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_entailment_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, self.pretrainer_params)
        pretrainer.train()

    def test_entailment_pretraining_does_not_crash_with_multiple_true_false_memory_network_solver(self):
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver)
        pretrainer = SnliEntailmentPretrainer(solver, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_memory_network_solver(self):
        solver = get_solver(MemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_multiple_true_false_memory_network_solver(self):
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_question_answer_memory_network_solver(self):
        solver = get_solver(QuestionAnswerMemoryNetworkSolver)
        pretrainer = SnliAttentionPretrainer(solver, self.pretrainer_params)
        pretrainer.train()

    def test_solver_training_works_after_pretraining(self):
        # TODO(matt): It's possible in this test that the pretrainers don't actually get called,
        # and we wouldn't know it.  Not sure how to make sure that the pretrainers are actually
        # called in this test.  You could probably do it with some complicated use of patching...
        args = {
                'pretrainers': [
                        {
                                'type': 'SnliEntailmentPretrainer',
                                'num_epochs': 1,
                                'train_files': [SNLI_FILE]
                                },
                        {
                                'type': 'SnliAttentionPretrainer',
                                'num_epochs': 1,
                                'train_files': [SNLI_FILE]
                                },
                        ]
                }
        solver = get_solver(MemoryNetworkSolver, args)
        solver.train()
