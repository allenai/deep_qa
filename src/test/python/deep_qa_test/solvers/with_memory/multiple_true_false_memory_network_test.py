# pylint: disable=no-self-use,invalid-name

from unittest import TestCase, mock
import os
import shutil

from deep_qa.solvers.with_memory.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetworkSolver
from ...common.constants import TEST_DIR
from ...common.solvers import get_solver
from ...common.solvers import write_multiple_true_false_memory_network_files
from ...common.test_markers import requires_tensorflow

class TestMultipleTrueFalseMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_multiple_true_false_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash(self):
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver)
        solver.train()

    @mock.patch.object(MultipleTrueFalseMemoryNetworkSolver, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        args = {
                'num_options': 5,
                'embedding_size': 2,
                'max_knowledge_length': 3,
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'answer_option_softmax',
                                'timedist_knowledge_selector_0',
                                ],
                        }
                }
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check in here that the attentions and so on are properly masked.  In
            # particular, we'll check two things: (1) that the final answer option softmax has
            # correctly padded out the extra option, and (2) that the attention weights on all of
            # the inputs are properly padded.  To see that this test is correct, you have to look
            # at the actual file written in `write_multiple_true_false_memory_network_files()`.
            answer_scores = output_dict['answer_option_softmax'][0]
            assert answer_scores[4] == 0
            attention_weights = output_dict['timedist_knowledge_selector_0'][0]
            assert attention_weights[0][2] == 0
            assert attention_weights[1][1] == 0
            assert attention_weights[1][2] == 0
            assert attention_weights[2][1] == 0
            assert attention_weights[2][2] == 0
            assert attention_weights[3][2] == 0
            assert attention_weights[4][0] == 0
            assert attention_weights[4][1] == 0
            assert attention_weights[4][2] == 0
        _output_debug_info.side_effect = new_debug
        solver.train()

    @mock.patch.object(MultipleTrueFalseMemoryNetworkSolver, '_output_debug_info')
    def test_works_with_words_and_characters_encoder(self, _output_debug_info):
        args = {
                'embedding_size': 4,
                'max_knowledge_length': 3,
                'tokenizer': {'type': 'words and characters'},
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'combined_word_embedding_for_background_input',
                                ],
                        'masks': [
                                'combined_word_embedding_for_background_input',
                                ],
                        }
                }
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check two things in here: that the shape of combined word embedding is
            # as expected, and that the mask is computed correctly.
            # TODO(matt): actually, from this test, it looks like the mask is returned as
            # output_dict['combined_word_embedding'][1].  Maybe this means we can simplify the
            # logic in Trainer._debug()?  I need to look into this more to be sure that's
            # consistently happening, though.
            word_embeddings = output_dict['combined_word_embedding_for_background_input'][0]
            assert len(word_embeddings) == 1
            assert word_embeddings[0].shape == (4, 3, 1, 4)
            word_masks = output_dict['masks']['combined_word_embedding_for_background_input'][0]
            print(word_masks)
            print(word_masks.shape)
            # Zeros are added to background sentences _from the right_.
            assert word_masks[0, 0, 0] == 1
            assert word_masks[0, 1, 0] == 1
            assert word_masks[0, 2, 0] == 0
            assert word_masks[1, 0, 0] == 1
            assert word_masks[1, 1, 0] == 0
            assert word_masks[1, 2, 0] == 0
            assert word_masks[2, 0, 0] == 1
            assert word_masks[2, 1, 0] == 0
            assert word_masks[2, 2, 0] == 0
            assert word_masks[3, 0, 0] == 1
            assert word_masks[3, 1, 0] == 1
            assert word_masks[3, 2, 0] == 0
        _output_debug_info.side_effect = new_debug
        solver.train()

    @requires_tensorflow
    def test_train_does_not_crash_using_adaptive_recurrence(self):
        args = {'recurrence_mode': {'type': 'adaptive'}}
        solver = get_solver(MultipleTrueFalseMemoryNetworkSolver, args)
        solver.train()
