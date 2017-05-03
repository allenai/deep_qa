import json
import os

from deep_qa.models.text_classification import ClassificationModel
from deep_qa.run import run_model, load_model, evaluate_model

from ..common.test_case import DeepQaTestCase


class TestRunModel(DeepQaTestCase):

    def setUp(self):
        super(TestRunModel, self).setUp()
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

    def test_evaluate_model(self):
        run_model(self.param_path)
        evaluate_model(self.param_path)
