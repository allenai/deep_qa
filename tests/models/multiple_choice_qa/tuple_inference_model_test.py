# pylint: disable=no-self-use,invalid-name
from numpy.testing.utils import assert_allclose

from deep_qa.models.multiple_choice_qa.tuple_inference import TupleInferenceModel
from ...common.test_case import DeepQaTestCase

class TestTupleInferenceModel(DeepQaTestCase):
    def test_model_trains_and_loads_correctly(self):
        self.write_tuple_inference_files()
        args = {"tuple_matcher": {"num_hidden_layers": 1,
                                  "hidden_layer_width": 4,
                                  "hidden_layer_activation": "tanh"},
                "num_question_tuples": 5,
                "num_background_tuples": 5,
                "num_tuple_slots": 4,
                "num_sentence_words": 10,
                "num_answer_options": 4,
                "save_models": True}
        solver = self.get_model(TupleInferenceModel, args)
        solver.train()

        loaded_model = self.get_model(TupleInferenceModel)
        loaded_model.load_model()

        assert_allclose(solver.model.predict(solver.__dict__["validation_input"]),
                        loaded_model.model.predict(solver.__dict__["validation_input"]),
                        rtol=1e-5)
