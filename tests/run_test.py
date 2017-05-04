import json
import os

from deep_qa.models.text_classification import ClassificationModel
from deep_qa.run import run_model, load_model, evaluate_model, score_dataset

from .common.test_case import DeepQaTestCase


class TestRun(DeepQaTestCase):
    # Our point here is mostly just to make sure the scripts don't crash.
    def setUp(self):
        super(TestRun, self).setUp()
        self.write_true_false_model_files()
        model_params = self.get_model_params(ClassificationModel, {"model_class": "ClassificationModel",
                                                                   'save_models': True})
        self.param_path = os.path.join(self.TEST_DIR, "params.json")
        with open(self.param_path, "w") as file_path:
            json.dump(model_params.as_dict(), file_path)

    def test_run_model(self):
        run_model(self.param_path)

    def test_load_model(self):
        run_model(self.param_path)
        loaded_model = load_model(self.param_path)
        assert loaded_model.can_train()

    def test_score_dataset(self):
        run_model(self.param_path)
        score_dataset(self.param_path)

    def test_evalaute_model(self):
        run_model(self.param_path)
        evaluate_model(self.param_path, [self.TEST_FILE])
