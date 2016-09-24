# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import argparse
import codecs
import os
import shutil

from dlfa.solvers.memory_network import MemoryNetworkSolver


class TestMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.test_dir = './TMP_TEST/'
        os.mkdir(self.test_dir)
        self.validation_file = self.test_dir + 'validation_file'
        with codecs.open(self.validation_file, 'w', 'utf-8') as validation_file:
            validation_file.write('1\tq1a1\t0\n')
            validation_file.write('2\tq1a2\t1\n')
            validation_file.write('3\tq1a3\t0\n')
            validation_file.write('4\tq1a4\t0\n')
        self.validation_background = self.test_dir + 'validation_background'
        with codecs.open(self.validation_background, 'w', 'utf-8') as validation_background:
            validation_background.write('1\tvb1\tvb2\n')
            validation_background.write('2\tvb3\tvb4\tvb5\n')
            validation_background.write('3\tvb6\n')
            validation_background.write('4\tvb7\tvb8\tvb9\n')
        self.train_file = self.test_dir + 'train_file'
        with codecs.open(self.train_file, 'w', 'utf-8') as train_file:
            train_file.write('1\tsentence1\t0\n')
            train_file.write('2\tsentence2\t1\n')
            train_file.write('3\tsentence3\t0\n')
            train_file.write('4\tsentence4\t1\n')
            train_file.write('5\tsentence5\t0\n')
            train_file.write('6\tsentence6\t0\n')
        self.train_background = self.test_dir + 'train_background'
        with codecs.open(self.train_background, 'w', 'utf-8') as train_background:
            train_background.write('1\tsb1\tsb2\n')
            train_background.write('2\tsb3\n')
            train_background.write('3\tsb4\n')
            train_background.write('4\tsb5\tsb6\n')
            train_background.write('5\tsb7\tsb8\n')
            train_background.write('6\tsb9\n')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_solver(self, additional_arguments=None):
        # We use the argument parser because it's easiest to get default values for all of the
        # parameters this way.
        parser = argparse.ArgumentParser()
        MemoryNetworkSolver.update_arg_parser(parser)
        arguments = {}
        arguments['model_serialization_prefix'] = self.test_dir
        arguments['train_file'] = self.train_file
        arguments['train_background'] = self.train_background
        arguments['validation_file'] = self.validation_file
        arguments['validation_background'] = self.validation_background
        arguments['embedding_size'] = '5'
        arguments['encoder'] = 'bow'
        arguments['knowledge_selector'] = 'dot_product'
        arguments['memory_updater'] = 'sum'
        arguments['entailment_input_combiner'] = 'memory_only'
        arguments['entailment_model'] = 'basic_mlp'
        arguments['num_epochs'] = '1'
        arguments['keras_validation_split'] = '0.0'
        if additional_arguments:
            for key, value in additional_arguments.items():
                arguments[key] = value
        argument_list = []
        for key, value in arguments.items():
            argument_list.append('--' + key)
            argument_list.append(value)
        args = parser.parse_args(argument_list)
        return MemoryNetworkSolver(**vars(args))

    def test_train_does_not_crash(self):
        solver = self._get_solver()
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': 'parameterized'}
        solver = self._get_solver(args)
        solver.train()

    def test_train_does_not_crash_with_cnn_encoder(self):
        args = {'encoder': 'cnn', 'cnn_ngram_filter_sizes': '1', 'cnn_num_filters': '1'}
        solver = self._get_solver(args)
        solver.train()
