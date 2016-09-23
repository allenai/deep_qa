# pylint: disable=invalid-name
import argparse
import codecs

from dlfa.solvers.memory_network import MemoryNetworkSolver
from dlfa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from dlfa.solvers.differentiable_search import DifferentiableSearchSolver

from .constants import TEST_DIR
from .constants import TRAIN_FILE
from .constants import TRAIN_BACKGROUND
from .constants import VALIDATION_FILE
from .constants import VALIDATION_BACKGROUND
from .constants import SNLI_FILE

def get_solver(cls, additional_arguments=None):
    parser = argparse.ArgumentParser()
    cls.update_arg_parser(parser)
    arguments = {}
    arguments['model_serialization_prefix'] = TEST_DIR
    arguments['train_file'] = TRAIN_FILE
    arguments['validation_file'] = VALIDATION_FILE
    arguments['embedding_size'] = '5'
    arguments['encoder'] = 'bow'
    arguments['num_epochs'] = '1'
    arguments['keras_validation_split'] = '0.0'
    if is_memory_network_solver(cls):
        arguments['train_background'] = TRAIN_BACKGROUND
        arguments['validation_background'] = VALIDATION_BACKGROUND
        arguments['knowledge_selector'] = 'dot_product'
        arguments['memory_updater'] = 'sum'
        arguments['entailment_input_combiner'] = 'memory_only'
    if additional_arguments:
        for key, value in additional_arguments.items():
            arguments[key] = value
    argument_list = []
    for key, value in arguments.items():
        argument_list.append('--' + key)
        if value is not None:
            argument_list.append(value)
    args = parser.parse_args(argument_list)
    return cls(**vars(args))


def write_snli_file():
    with codecs.open(SNLI_FILE, 'w', 'utf-8') as snli_file:
        snli_file.write('1\ttext1\thypothesis1\tentails\n')
        snli_file.write('2\ttext2\thypothesis2\tcontradicts\n')
        snli_file.write('3\ttext3\thypothesis3\tentails\n')
        snli_file.write('4\ttext4\thypothesis4\tneutral\n')
        snli_file.write('5\ttext5\thypothesis5\tentails\n')
        snli_file.write('6\ttext6\thypothesis6\tcontradicts\n')


def write_memory_network_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as validation_file:
        validation_file.write('1\tq1a1\t0\n')
        validation_file.write('2\tq1a2\t1\n')
        validation_file.write('3\tq1a3\t0\n')
        validation_file.write('4\tq1a4\t0\n')
    with codecs.open(VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
        validation_background.write('1\tvb1\tvb2\n')
        validation_background.write('2\tvb3\tvb4\tvb5\n')
        validation_background.write('3\tvb6\n')
        validation_background.write('4\tvb7\tvb8\tvb9\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tsentence1\t0\n')
        train_file.write('2\tsentence2\t1\n')
        train_file.write('3\tsentence3\t0\n')
        train_file.write('4\tsentence4\t1\n')
        train_file.write('5\tsentence5\t0\n')
        train_file.write('6\tsentence6\t0\n')
    with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
        train_background.write('1\tsb1\tsb2\n')
        train_background.write('2\tsb3\n')
        train_background.write('3\tsb4\n')
        train_background.write('4\tsb5\tsb6\n')
        train_background.write('5\tsb7\tsb8\n')
        train_background.write('6\tsb9\n')


def write_multiple_choice_memory_network_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as validation_file:
        validation_file.write('1\tq1a1\t0\n')
        validation_file.write('2\tq1a2\t1\n')
        validation_file.write('3\tq1a3\t0\n')
        validation_file.write('4\tq1a4\t0\n')
    with codecs.open(VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
        validation_background.write('1\tvb1\tvb2\n')
        validation_background.write('2\tvb3\tvb4\tvb5\n')
        validation_background.write('3\tvb6\n')
        validation_background.write('4\tvb7\tvb8\tvb9\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tsentence1\t0\n')
        train_file.write('2\tsentence2\t0\n')
        train_file.write('3\tsentence3\t0\n')
        train_file.write('4\tsentence4\t1\n')
    with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
        train_background.write('1\tsb1\tsb2\n')
        train_background.write('2\tsb3\n')
        train_background.write('3\tsb4\n')
        train_background.write('4\tsb5\tsb6\n')


def write_question_answer_memory_network_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tquestion1\tanswer1###answer2\t0\n')
    with codecs.open(VALIDATION_BACKGROUND, 'w', 'utf-8') as validation_background:
        validation_background.write('1\tvb1\tvb2\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\ta b e i d\tanswer1###answer2\t0\n')
        train_file.write('2\ta b c d\tanswer3###answer4\t1\n')
        train_file.write('3\te d w f d s a\tanswer5###answer6###answer9\t2\n')
        train_file.write('4\te fj k w q\tanswer7###answer8\t0\n')
    with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
        train_background.write('1\tsb1\tsb2\n')
        train_background.write('2\tsb3\n')
        train_background.write('3\tsb4\n')
        train_background.write('4\tsb5\tsb6\n')


def is_memory_network_solver(cls):
    # pylint: disable=multiple-statements
    # TODO(matt): figure out how to do this with a call to isinstance()
    if cls == MemoryNetworkSolver: return True
    if cls == MultipleChoiceMemoryNetworkSolver: return True
    if cls == QuestionAnswerMemoryNetworkSolver: return True
    if cls == DifferentiableSearchSolver: return True
    return False
