# pylint: disable=no-self-use
import gzip

from deep_qa.models.memory_networks import DifferentiableSearchMemoryNetwork
from deep_qa.data.instances.text_classification import TextClassificationInstance
from deep_qa.common.params import Params
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
        args = Params({
                'corpus_path': self.corpus_path,
                'model_serialization_prefix': './',
                'num_sentence_words': 3,
                })
        model = self.get_model(DifferentiableSearchMemoryNetwork, args)
        model.encoder_model = FakeEncoder()
        model._initialize_lsh()

    def test_get_nearest_neighbors_does_not_crash(self):
        args = Params({
                'corpus_path': self.corpus_path,
                'model_serialization_prefix': './',
                'num_sentence_words': 5,
                })
        model = self.get_model(DifferentiableSearchMemoryNetwork, args)
        model.encoder_model = FakeEncoder()
        model._initialize_lsh()
        model.num_sentence_words = 5
        model.max_knowledge_length = 2
        model.get_nearest_neighbors(TextClassificationInstance("this is a sentence", True))
