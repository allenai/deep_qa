# pylint: disable=no-self-use,invalid-name
from unittest import TestCase
import os
import shutil

from numpy.testing.utils import assert_allclose

from deep_qa.models.multiple_choice_qa.tuple_inference import TupleInferenceModel
from ...common.constants import TEST_DIR
from ...common.models import get_model, write_tuple_inference_files

class TestTupleInferenceModel(TestCase):
    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_tuple_inference_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_model_trains_and_loads_correctly(self):
        args = {"tuple_matcher": {"num_hidden_layers": 1,
                                  "hidden_layer_width": 4,
                                  "hidden_layer_activation": "tanh"},
                "num_question_tuples": 5,
                "num_background_tuples": 5,
                "num_tuple_slots": 4,
                "word_sequence_length": 10,
                "num_answer_options": 4,
                "save_models": True}
        solver = get_model(TupleInferenceModel, args)
        solver.train()

        loaded_model = get_model(TupleInferenceModel)
        loaded_model.load_model()

        assert_allclose(solver.model.predict(solver.__dict__["validation_input"]),
                        loaded_model.model.predict(solver.__dict__["validation_input"]))
