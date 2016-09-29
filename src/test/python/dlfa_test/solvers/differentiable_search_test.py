# pylint: disable=no-self-use
import gzip
import os

from unittest import TestCase

from dlfa.solvers.differentiable_search import DifferentiableSearchSolver
from dlfa.data.text_instance import TrueFalseInstance
from ..common.solvers import get_solver

class FakeEncoder:
    def predict(self, instances):
        predictions = []
        for _ in instances:
            predictions.append([1.0, -1.0, 0.5, 0.25])
        return predictions

class TestDifferentiableSearchSolver(TestCase):
    # pylint: disable=protected-access
    def setUp(self):
        self.corpus_path = './TMP_FAKE_corpus.gz'
        with gzip.open(self.corpus_path, 'wb') as corpus_file:
            corpus_file.write('this is a sentence\n'.encode('utf-8'))
            corpus_file.write('this is another sentence\n'.encode('utf-8'))
            corpus_file.write('a really great sentence\n'.encode('utf-8'))
            corpus_file.write('scientists study animals\n'.encode('utf-8'))

    def tearDown(self):
        os.remove(self.corpus_path)

    def test_initialize_lsh_does_not_crash(self):
        args = {'corpus_path': self.corpus_path, 'model_serialization_prefix': './'}
        solver = get_solver(DifferentiableSearchSolver, args)
        solver.encoder_model = FakeEncoder()
        solver._initialize_lsh()

    def test_get_nearest_neighbors_does_not_crash(self):
        args = {'corpus_path': self.corpus_path, 'model_serialization_prefix': './'}
        solver = get_solver(DifferentiableSearchSolver, args)
        solver.encoder_model = FakeEncoder()
        solver._initialize_lsh()
        solver.max_sentence_length = 5
        solver.max_knowledge_length = 2
        solver.get_nearest_neighbors(TrueFalseInstance("this is a sentence", True))
