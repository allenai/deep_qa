# pylint: disable=no-self-use
import argparse
import gzip

from pyfakefs import fake_filesystem_unittest

from dlfa.solvers.differentiable_search import DifferentiableSearchSolver
from dlfa.data.instance import TextInstance

class FakeEncoder:
    def predict(self, instances):
        predictions = []
        for _ in instances:
            predictions.append([1.0, -1.0, 0.5, 0.25])
        return predictions

class TestDifferentiableSearchSolver(fake_filesystem_unittest.TestCase):
    # pylint: disable=protected-access
    def setUp(self):
        self.setUpPyfakefs()
        self.corpus_path = '/corpus.gz'
        with gzip.open(self.corpus_path, 'wb') as corpus_file:
            corpus_file.write('this is a sentence\n'.encode('utf-8'))
            corpus_file.write('this is another sentence\n'.encode('utf-8'))
            corpus_file.write('a really great sentence\n'.encode('utf-8'))
            corpus_file.write('scientists study animals\n'.encode('utf-8'))

    def get_solver(self) -> DifferentiableSearchSolver:
        # It's easiest to get default values for all of the parameters this way.
        parser = argparse.ArgumentParser()
        DifferentiableSearchSolver.update_arg_parser(parser)
        additional_arguments = ['--corpus_path', self.corpus_path,
                                '--model_serialization_prefix', '.']
        args = parser.parse_args(additional_arguments)
        return DifferentiableSearchSolver(**vars(args))

    def test_initialize_lsh_does_not_crash(self):
        solver = self.get_solver()
        solver.encoder_model = FakeEncoder()
        solver._initialize_lsh()

    def test_get_nearest_neighbors_does_not_crash(self):
        solver = self.get_solver()
        solver.encoder_model = FakeEncoder()
        solver._initialize_lsh()
        solver.max_sentence_length = 5
        solver.max_knowledge_length = 2
        solver.get_nearest_neighbors(TextInstance("this is a sentence", True))
