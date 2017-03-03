# pylint: disable=no-self-use
import gzip

from deep_qa.models.memory_networks.differentiable_search import DifferentiableSearchMemoryNetwork
from deep_qa.data.instances.true_false_instance import TrueFalseInstance
from ...common.test_case import DeepQaTestCase

class FakeEncoder:
    def predict(self, instances):
        predictions = []
        for _ in instances:
            predictions.append([1.0, -1.0, 0.5, 0.25])
        return predictions

class TestDifferentiableSearchMemoryNetwork(DeepQaTestCase):
    # pylint: disable=protected-access
    def setUp(self):
        super(TestDifferentiableSearchMemoryNetwork, self).setUp()
        self.corpus_path = self.TEST_DIR + 'FAKE_corpus.gz'
        with gzip.open(self.corpus_path, 'wb') as corpus_file:
            corpus_file.write('this is a sentence\n'.encode('utf-8'))
            corpus_file.write('this is another sentence\n'.encode('utf-8'))
            corpus_file.write('a really great sentence\n'.encode('utf-8'))
            corpus_file.write('scientists study animals\n'.encode('utf-8'))

    def test_initialize_lsh_does_not_crash(self):
        args = {
                'corpus_path': self.corpus_path,
                'model_serialization_prefix': './',
                'max_sentence_length': 3,
                }
        model = self.get_model(DifferentiableSearchMemoryNetwork, args)
        model.encoder_model = FakeEncoder()
        model._initialize_lsh()

    def test_get_nearest_neighbors_does_not_crash(self):
        args = {
                'corpus_path': self.corpus_path,
                'model_serialization_prefix': './',
                'max_sentence_length': 5,
                }
        model = self.get_model(DifferentiableSearchMemoryNetwork, args)
        model.encoder_model = FakeEncoder()
        model._initialize_lsh()
        model.max_sentence_length = 5
        model.max_knowledge_length = 2
        model.get_nearest_neighbors(TrueFalseInstance("this is a sentence", True))
