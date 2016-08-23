# pylint: disable=no-self-use
import argparse
import gzip

from typing import List
from unittest.mock import patch

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
    def setUp(self):
        self.setUpPyfakefs()

    def get_solver(self, additional_arguments: List[str]) -> DifferentiableSearchSolver:
        # It's easiest to get default values for all of the parameters this way.
        parser = argparse.ArgumentParser()
        DifferentiableSearchSolver.update_arg_parser(parser)
        args = parser.parse_args(additional_arguments)
        return DifferentiableSearchSolver(**vars(args))

    def mock_words(self):
        return ['these', 'are', 'fake', 'words']

    # We need to mock out TextInstance.words(), because it calls nltk, which needs access to the
    # real filesystem...
    @patch.object(TextInstance, 'words', mock_words)
    def test_initialize_lsh_does_not_crash(self):
        # pylint: disable=protected-access
        corpus_path = '/corpus.gz'
        with gzip.open(corpus_path, 'wb') as corpus_file:
            corpus_file.write('this is a sentence\n'.encode('utf-8'))
            corpus_file.write('this is another sentence\n'.encode('utf-8'))
            corpus_file.write('a really great sentence\n'.encode('utf-8'))
            corpus_file.write('scientists study animals\n'.encode('utf-8'))
        additional_arguments = ['--corpus_path', corpus_path, '--model_serialization_prefix', '.']
        solver = self.get_solver(additional_arguments)
        solver.encoder_model = FakeEncoder()
        solver._initialize_lsh()
