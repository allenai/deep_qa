import argparse

from dlfa.solvers.memory_network import MemoryNetworkSolver
from dlfa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from dlfa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from dlfa.solvers.differentiable_search import DifferentiableSearchSolver

from .constants import TEST_DIR
from .constants import TRAIN_FILE
from .constants import TRAIN_BACKGROUND
from .constants import VALIDATION_FILE
from .constants import VALIDATION_BACKGROUND

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
        argument_list.append(value)
    args = parser.parse_args(argument_list)
    return cls(**vars(args))


def is_memory_network_solver(cls):
    # pylint: disable=multiple-statements
    # TODO(matt): figure out how to do this with a call to isinstance()
    if cls == MemoryNetworkSolver: return True
    if cls == MultipleChoiceMemoryNetworkSolver: return True
    if cls == QuestionAnswerMemoryNetworkSolver: return True
    if cls == DifferentiableSearchSolver: return True
    return False
