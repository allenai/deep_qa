# pylint: disable=no-self-use,invalid-name
from deep_qa.models import MultipleTrueFalseSimilarity
from ...common.test_case import DeepQaTestCase
from ...common.test_markers import requires_tensorflow


class TestMultipleTrueFalseSimilarity(DeepQaTestCase):
    @requires_tensorflow
    def test_train_does_not_crash(self):
        self.write_multiple_true_false_memory_network_files()
        model = self.get_model(MultipleTrueFalseSimilarity)
        model.train()
