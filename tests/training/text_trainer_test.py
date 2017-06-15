# pylint: disable=no-self-use,invalid-name
from unittest import mock

from deep_qa.common.params import Params, pop_choice
from deep_qa.data.datasets import Dataset, SnliDataset
from deep_qa.layers.encoders import encoders
from deep_qa.models.text_classification import ClassificationModel
from deep_qa.testing.test_case import DeepQaTestCase


class TestTextTrainer(DeepQaTestCase):
    # pylint: disable=protected-access

    def test_get_encoder_works_without_params(self):
        self.write_true_false_model_files()
        model = self.get_model(ClassificationModel, {'encoder': {}})
        model._build_model()
        encoder = model._get_encoder()
        encoder_type = pop_choice({}, "type", list(encoders.keys()), default_to_first_choice=True)
        expected_encoder = encoders[encoder_type](**{})
        assert isinstance(encoder, expected_encoder.__class__)

    @mock.patch.object(ClassificationModel, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        self.write_true_false_model_files()
        args = Params({
                'embeddings': {'words': {'dimension': 2}, 'characters': {'dimension': 2}},
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
                })
        model = self.get_model(ClassificationModel, args)

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
            word_masks = output_dict['combined_word_embedding_for_sentence_input'][1]
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

    def test_load_model_and_fit(self):
        args = Params({
                'test_files': [self.TEST_FILE],
                'embeddings': {'words': {'dimension': 4}, 'characters': {'dimension': 2}},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
        })
        self.write_true_false_model_files()
        model, loaded_model = self.ensure_model_trains_and_loads(ClassificationModel, args)

        # now fit both models on some more data, and ensure that we get the same results.
        self.write_additional_true_false_model_files()
        _, training_arrays = loaded_model.load_data_arrays(loaded_model.train_files)
        model.model.fit(training_arrays[0], training_arrays[1], shuffle=False, nb_epoch=1)
        loaded_model.model.fit(training_arrays[0], training_arrays[1], shuffle=False, nb_epoch=1)

        # _, validation_arrays = loaded_model.load_data_arrays(loaded_model.validation_files)
        # verify that original model and the loaded model predict the same outputs
        # TODO(matt): fix the randomness that occurs here.
        # assert_allclose(model.model.predict(validation_arrays[0]),
        #                 loaded_model.model.predict(validation_arrays[0]))

    def test_data_generator_works(self):
        args = Params({
                'test_files': [self.TEST_FILE],
                'embeddings': {'words': {'dimension': 4}, 'characters': {'dimension': 2}},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'data_generator': {},
                'show_summary_with_masking_info': True,
        })
        self.write_true_false_model_files()
        self.ensure_model_trains_and_loads(ClassificationModel, args)

    def test_dynamic_padding_works(self):
        args = Params({
                'test_files': [self.TEST_FILE],
                'embeddings': {'words': {'dimension': 4}, 'characters': {'dimension': 2}},
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'data_generator': {'dynamic_padding': True},
                'batch_size': 2,
        })
        self.write_true_false_model_files()
        self.ensure_model_trains_and_loads(ClassificationModel, args)

    def test_pretrained_embeddings_works_correctly(self):
        self.write_true_false_model_files()
        self.write_pretrained_vector_files()
        args = Params({
                'embeddings': {
                        'words': {
                                'dimension': 8,
                                'pretrained_file': self.PRETRAINED_VECTORS_GZIP,
                                'project': True
                        },
                        'characters': {'dimension': 8}},
                })
        model = self.get_model(ClassificationModel, args)
        model.train()

    def test_reading_two_datasets_return_identical_types(self):

        self.write_true_false_model_files()
        model = self.get_model(ClassificationModel)
        train_dataset = model.load_dataset_from_files([self.TRAIN_FILE])
        validation_dataset = model.load_dataset_from_files([self.VALIDATION_FILE])

        assert isinstance(train_dataset, Dataset)
        assert isinstance(validation_dataset, Dataset)

    def test_reading_two_non_default_datasets_return_identical_types(self):

        self.write_original_snli_data()
        model = self.get_model(ClassificationModel, {"dataset": {"type": "snli"}})
        train_dataset = model.load_dataset_from_files([self.TRAIN_FILE])
        validation_dataset = model.load_dataset_from_files([self.TRAIN_FILE])

        assert isinstance(train_dataset, SnliDataset)
        assert isinstance(validation_dataset, SnliDataset)
