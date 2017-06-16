# pylint: disable=invalid-name,no-self-use
import json
import os

import numpy
from numpy.testing import assert_almost_equal
from deep_qa.run import compute_accuracy
from deep_qa.run import run_model_from_file, load_model, evaluate_model
from deep_qa.run import score_dataset, score_dataset_with_ensemble
from deep_qa.testing.test_case import DeepQaTestCase


class TestRun(DeepQaTestCase):
    # Our point here is mostly just to make sure the scripts don't crash.
    def setUp(self):
        super(TestRun, self).setUp()
        self.write_true_false_model_files()
        model_params = self.get_model_params({"model_class": "ClassificationModel",
                                              'save_models': True})
        self.param_path = os.path.join(self.TEST_DIR, "params.json")
        with open(self.param_path, "w") as file_path:
            json.dump(model_params.as_dict(), file_path)

    def test_run_model_does_not_crash(self):
        run_model_from_file(self.param_path)

    def test_load_model_does_not_crash(self):
        run_model_from_file(self.param_path)
        loaded_model = load_model(self.param_path)
        assert loaded_model.can_train()

    def test_score_dataset_does_not_crash(self):
        run_model_from_file(self.param_path)
        score_dataset(self.param_path, [self.TEST_FILE])

    def test_evalaute_model_does_not_crash(self):
        run_model_from_file(self.param_path)
        evaluate_model(self.param_path, [self.TEST_FILE])

    def test_score_dataset_with_ensemble_gives_same_predictions_as_score_dataset(self):
        # We're just going to test something simple here: that the methods don't crash, and that we
        # get the same result with an ensemble of one model that we do with `score_dataset`.
        run_model_from_file(self.param_path)
        predictions, _ = score_dataset(self.param_path, [self.TEST_FILE])
        ensembled_predictions, _ = score_dataset_with_ensemble([self.param_path], [self.TEST_FILE])
        assert_almost_equal(predictions, ensembled_predictions)

    def test_compute_accuracy_computes_a_correct_metric(self):
        predictions = numpy.asarray([[.5, .5, .6], [.1, .4, .0]])
        labels = numpy.asarray([[1, 0, 0], [0, 1, 0]])
        assert compute_accuracy(predictions, labels) == .5
