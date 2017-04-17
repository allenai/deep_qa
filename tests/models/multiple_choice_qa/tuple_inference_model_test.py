# pylint: disable=no-self-use,invalid-name
from deep_qa.models import TupleInferenceModel
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
        self.ensure_model_trains_and_loads(TupleInferenceModel, args)
