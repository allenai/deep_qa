# pylint: disable=no-self-use,invalid-name

from unittest import TestCase, mock
import os
import shutil

import numpy

from deep_qa.models.text_classification.true_false_model import TrueFalseModel
from deep_qa.models.multiple_choice_qa.question_answer_similarity import QuestionAnswerSimilarity
from ..common.constants import TEST_DIR
from ..common.models import get_model
from ..common.models import write_question_answer_memory_network_files
from ..common.models import write_true_false_model_files
from ..common.test_markers import requires_tensorflow


class TestTextTrainer(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_true_false_model_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    @mock.patch.object(TrueFalseModel, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        write_true_false_model_files()
        args = {
                'embedding_size': 4,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'combined_word_embedding_for_sentence_input',
                                ],
                        'masks': [
                                'combined_word_embedding_for_sentence_input',
                                ],
                        }
                }
        model = get_model(TrueFalseModel, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check two things in here: that the shape of combined word embedding is
            # as expected, and that the mask is computed correctly.
            # TODO(matt): actually, from this test, it looks like the mask is returned as
            # output_dict['combined_word_embedding'][1].  Maybe this means we can simplify the
            # logic in Trainer._debug()?  I need to look into this more to be sure that's
            # consistently happening, though.
            word_embeddings = output_dict['combined_word_embedding_for_sentence_input'][0]
            assert len(word_embeddings) == 6
            assert word_embeddings[0].shape == (3, 4)
            word_masks = output_dict['masks']['combined_word_embedding_for_sentence_input']
            # Zeros are added to sentences _from the left_.
            assert word_masks[0][0] == 0
            assert word_masks[0][1] == 0
            assert word_masks[0][2] == 1
            assert word_masks[1][0] == 1
            assert word_masks[1][1] == 1
            assert word_masks[1][2] == 1
            assert word_masks[2][0] == 0
            assert word_masks[2][1] == 1
            assert word_masks[2][2] == 1
            assert word_masks[3][0] == 0
            assert word_masks[3][1] == 0
            assert word_masks[3][2] == 1
        _output_debug_info.side_effect = new_debug
        model.train()

    @mock.patch.object(QuestionAnswerSimilarity, '_output_debug_info')
    def test_words_and_characters_works_with_matrices(self, _output_debug_info):
        write_question_answer_memory_network_files()
        args = {
                'embedding_size': 4,
                'tokenizer': {'type': 'words and characters'},
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'combined_word_embedding_for_answer_input',
                                ],
                        'masks': [
                                'combined_word_embedding_for_answer_input',
                                ],
                        }
                }
        model = get_model(QuestionAnswerSimilarity, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check two things in here: that the shape of combined word embedding is
            # as expected, and that the mask is computed correctly.
            # TODO(matt): actually, from this test, it looks like the mask is returned as
            # output_dict['combined_word_embedding'][1].  Maybe this means we can simplify the
            # logic in Trainer._debug()?  I need to look into this more to be sure that's
            # consistently happening, though.
            word_embeddings = output_dict['combined_word_embedding_for_answer_input'][0]
            assert len(word_embeddings) == 4
            assert word_embeddings[0].shape == (3, 2, 4)
            word_masks = output_dict['masks']['combined_word_embedding_for_answer_input']
            # Zeros are added to answer words _from the left_, and to answer options from the
            # _right_.
            assert numpy.all(word_masks[0, 0, :] == numpy.asarray([1, 1]))
            assert numpy.all(word_masks[0, 1, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[0, 2, :] == numpy.asarray([0, 0]))
            assert numpy.all(word_masks[1, 0, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[1, 1, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[1, 2, :] == numpy.asarray([0, 0]))
            assert numpy.all(word_masks[2, 0, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[2, 1, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[2, 2, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[3, 0, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[3, 1, :] == numpy.asarray([0, 1]))
            assert numpy.all(word_masks[3, 2, :] == numpy.asarray([0, 0]))
        _output_debug_info.side_effect = new_debug
        model.train()

    @requires_tensorflow
    def test_tensorboard_logs_does_not_crash(self):
        write_true_false_model_files()
        model = get_model(TrueFalseModel, {'tensorboard_log': TEST_DIR})
        model.train()
