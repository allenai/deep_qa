# pylint: disable=no-self-use,invalid-name

from unittest import TestCase, mock
import os
import shutil

from deep_qa.solvers.no_memory.question_answer_solver import QuestionAnswerSolver
from ...common.constants import TEST_DIR
from ...common.solvers import get_solver
from ...common.solvers import write_question_answer_memory_network_files


class TestQuestionAnswerSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_question_answer_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(QuestionAnswerSolver)
        solver.train()

    @mock.patch.object(QuestionAnswerSolver, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        args = {
                'num_hidden_layers': 1,
                'hidden_layer_width': 2,
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': ['answer_similarity_softmax']
                        }
                }
        solver = get_solver(QuestionAnswerSolver, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check in here that the attentions and so on are properly masked.  In
            # particular, we'll check two things: (1) that the final answer option softmax has
            # correctly padded out the extra option, and (2) that the attention weights on all of
            # the inputs are properly padded.  To see that this test is correct, you have to look
            # at the actual file written in `write_multiple_true_false_memory_network_files()`.
            print(output_dict)
            answer_scores = output_dict['answer_similarity_softmax']
            assert answer_scores[0][2] == 0
            assert answer_scores[1][2] == 0
            assert answer_scores[3][2] == 0
        _output_debug_info.side_effect = new_debug
        solver.train()
