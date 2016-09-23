# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import argparse
import codecs
import os
import shutil

from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver


class TestQuestionAnswerMemoryNetworkSolver(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.test_dir = './TMP_TEST/'
        os.mkdir(self.test_dir)
        self.validation_file = self.test_dir + 'validation_file'
        with codecs.open(self.validation_file, 'w', 'utf-8') as train_file:
            train_file.write('1\tquestion1\tanswer1###answer2\t0\n')
        self.validation_background = self.test_dir + 'validation_background'
        with codecs.open(self.validation_background, 'w', 'utf-8') as validation_background:
            validation_background.write('1\tvb1\tvb2\n')
        self.train_file = self.test_dir + 'train_file'
        with codecs.open(self.train_file, 'w', 'utf-8') as train_file:
            train_file.write('1\ta b e i d\tanswer1###answer2\t0\n')
            train_file.write('2\ta b c d\tanswer3###answer4\t1\n')
            train_file.write('3\te d w f d s a\tanswer5###answer6###answer9\t2\n')
            train_file.write('4\te fj k w q\tanswer7###answer8\t0\n')
        self.train_background = self.test_dir + 'train_background'
        with codecs.open(self.train_background, 'w', 'utf-8') as train_background:
            train_background.write('1\tsb1\tsb2\n')
            train_background.write('2\tsb3\n')
            train_background.write('3\tsb4\n')
            train_background.write('4\tsb5\tsb6\n')

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _get_solver(self, additional_arguments=None):
        # We use the argument parser because it's easiest to get default values for all of the
        # parameters this way.
        parser = argparse.ArgumentParser()
        QuestionAnswerMemoryNetworkSolver.update_arg_parser(parser)
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
        return QuestionAnswerMemoryNetworkSolver(**vars(args))

    def test_train_does_not_crash(self):
        solver = self._get_solver()
        solver.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': 'parameterized'}
        solver = self._get_solver(args)
        solver.train()
