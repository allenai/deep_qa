# pylint: disable=no-self-use,invalid-name
from unittest import mock

import numpy
from numpy.testing import assert_allclose

from deep_qa.common.params import get_choice_with_default
from deep_qa.layers.encoders import encoders
from deep_qa.models.text_classification.true_false_model import TrueFalseModel
from deep_qa.models.multiple_choice_qa.question_answer_similarity import QuestionAnswerSimilarity
from ..common.test_case import DeepQaTestCase


class TestTextTrainer(DeepQaTestCase):
    # pylint: disable=protected-access

    def test_get_encoder_works_without_params(self):
        self.write_true_false_model_files()
        model = self.get_model(TrueFalseModel, {'encoder': {}})
        encoder = model._get_encoder()
        encoder_type = get_choice_with_default({}, "type", list(encoders.keys()))
        expected_encoder = encoders[encoder_type](**{})
        assert isinstance(encoder, expected_encoder.__class__)

    @mock.patch.object(TrueFalseModel, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        self.write_true_false_model_files()
        args = {
                'embedding_dim': {'words': 2, 'characters': 2},
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
        model = self.get_model(TrueFalseModel, args)

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
        self.write_question_answer_memory_network_files()
        args = {
                'embedding_dim': {'words': 2, 'characters': 2},
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
        model = self.get_model(QuestionAnswerSimilarity, args)

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

    def test_load_model(self):
        # train a model and serialize it.
        args = {
                'embedding_dim': {'words': 4, 'characters': 4},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
        }
        self.write_true_false_model_files()
        model = self.get_model(TrueFalseModel, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(TrueFalseModel, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))

    def test_load_model_and_fit(self):
        # train a model and serialize it.
        args = {
                'embedding_dim': {'words': 4, 'characters': 4},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
        }
        self.write_true_false_model_files()
        model = self.get_model(TrueFalseModel, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(TrueFalseModel, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))

        # now fit both models on some more data, and ensure that we get the same results.
        self.write_additional_true_false_model_files()
        # pylint: disable=unused-variable
        train_data, val_data = loaded_model.prepare_data(loaded_model.train_files,
                                                         loaded_model.max_training_instances,
                                                         loaded_model.validation_files,
                                                         update_data_indexer=False)
        _, train_input, train_labels = train_data
        # _, validation_input, _ = val_data
        model.model.fit(train_input, train_labels, shuffle=False, nb_epoch=1)
        loaded_model.model.fit(train_input, train_labels, shuffle=False, nb_epoch=1)

        # verify that original model and the loaded model predict the same outputs
        # TODO(matt): fix the randomness that occurs here.
        # assert_allclose(model.model.predict(validation_input),
        #                 loaded_model.model.predict(validation_input))

    def test_pretrained_embeddings_works_correctly(self):
        self.write_true_false_model_files()
        self.write_pretrained_vector_files()
        args = {
                'embedding_dim': {'words': 8, 'characters': 8},
                'pretrained_embeddings_file': self.PRETRAINED_VECTORS_GZIP,
                'fine_tune_embeddings': False,
                'project_embeddings': False,
                }
        model = self.get_model(TrueFalseModel, args)
        model.train()
