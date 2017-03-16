# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_allclose
from flaky import flaky

from deep_qa.models.reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from ...common.test_case import DeepQaTestCase


class TestBidirectionalAttentionFlow(DeepQaTestCase):
    @flaky
    def test_trains_and_loads_correctly(self):
        self.write_span_prediction_files()
        args = {
                'embedding_dim': {'words': 4, 'characters': 4},
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

        # We should get the same result if we index the data from the
        # original model and the loaded model.
        indexed_validation_input, _ = loaded_model._prepare_data( # pylint: disable=protected-access
                model.__dict__["validation_dataset"],
                for_train=False)
        assert_allclose(model.model.predict(model.__dict__["validation_input"]),
                        loaded_model.model.predict(indexed_validation_input))

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

    def test_get_best_span(self):
        # Note that the best span cannot be (1, 0) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = numpy.array([0.1, 0.3, 0.05, 0.3, 0.25])
        span_end_probs = numpy.array([0.5, 0.1, 0.2, 0.05, 0.15])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (1, 2)

        # Testing an edge case of the dynamic program here, for the order of when you update the
        # best previous span position.  We should not get (1, 1), because that's an empty span.
        span_begin_probs = numpy.array([0.4, 0.5, 0.1])
        span_end_probs = numpy.array([0.3, 0.6, 0.1])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (0, 1)

        # test higher-order input
        # Note that the best span cannot be (1, 1) since even though 0.3 * 0.5 is the greatest
        # value, the end span index is constrained to occur after the begin span index.
        span_begin_probs = numpy.array([[0.1, 0.3, 0.05, 0.3, 0.25]])
        span_end_probs = numpy.array([[0.1, 0.5, 0.2, 0.05, 0.15]])
        begin_end_idxs = BidirectionalAttentionFlow.get_best_span(span_begin_probs,
                                                                  span_end_probs)
        assert begin_end_idxs == (1, 2)
