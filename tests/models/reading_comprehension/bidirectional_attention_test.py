# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_allclose

from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...common.test_case import DeepQaTestCase


class TestBidirectionalAttentionFlow(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = {
                'embedding_size': 4,
                'save_models': True,
                'tokenizer': {'type': 'words and characters'},
                'show_summary_with_masking_info': True,
                }
        model = self.get_model(BidirectionalAttentionFlow, args)
        model.train()

        # load the model that we serialized
        loaded_model = self.get_model(BidirectionalAttentionFlow, args)
        loaded_model.load_model()

        # verify that original model and the loaded model predict the same outputs
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(model.__dict__["validation_input"]))

        # now fit both models on some more data, and ensure that we get the same results.
        self.write_additional_span_prediction_files()
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
