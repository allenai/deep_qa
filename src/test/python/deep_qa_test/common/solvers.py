# pylint: disable=invalid-name
import codecs

from deep_qa.solvers.memory_network import MemoryNetworkSolver
from deep_qa.solvers.multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from deep_qa.solvers.multiple_choice_similarity import MultipleChoiceSimilaritySolver
from deep_qa.solvers.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from deep_qa.solvers.differentiable_search import DifferentiableSearchSolver

from .constants import TEST_DIR
from .constants import TRAIN_FILE
from .constants import TRAIN_BACKGROUND
from .constants import VALIDATION_FILE
from .constants import VALIDATION_BACKGROUND
from .constants import SNLI_FILE

def get_solver(cls, additional_arguments=None):
    params = {}
    params['model_serialization_prefix'] = TEST_DIR
    params['train_file'] = TRAIN_FILE
    params['validation_file'] = VALIDATION_FILE
    params['embedding_size'] = 5
    params['encoder'] = {'type': 'bow'}
    params['num_epochs'] = 1
    params['keras_validation_split'] = 0.0
    if is_solver_with_background(cls):
        params['train_background'] = TRAIN_BACKGROUND
        params['validation_background'] = VALIDATION_BACKGROUND
    if is_memory_network_solver(cls):
        params['knowledge_selector'] = {'type': 'dot_product'}
        params['memory_updater'] = {'type': 'sum'}
        params['entailment_input_combiner'] = {'type': 'memory_only'}
    if additional_arguments:
        for key, value in additional_arguments.items():
            params[key] = value
    return cls(params)


def write_snli_file():
    with codecs.open(SNLI_FILE, 'w', 'utf-8') as snli_file:
        snli_file.write('1\ttext1\thypothesis1\tentails\n')
        snli_file.write('2\ttext2\thypothesis2\tcontradicts\n')
        snli_file.write('3\ttext3\thypothesis3\tentails\n')
        snli_file.write('4\ttext4\thypothesis4\tneutral\n')
        snli_file.write('5\ttext5\thypothesis5\tentails\n')
        snli_file.write('6\ttext6\thypothesis6\tcontradicts\n')


def write_lstm_solver_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as validation_file:
        validation_file.write('1\tq1a1\t0\n')
        validation_file.write('2\tq1a2\t1\n')
        validation_file.write('3\tq1a3\t0\n')
        validation_file.write('4\tq1a4\t0\n')
        validation_file.write('5\tq2a1\t0\n')
        validation_file.write('6\tq2a2\t0\n')
        validation_file.write('7\tq2a3\t1\n')
        validation_file.write('8\tq2a4\t0\n')
        validation_file.write('9\tq3a1\t0\n')
        validation_file.write('10\tq3a2\t0\n')
        validation_file.write('11\tq3a3\t0\n')
        validation_file.write('12\tq3a4\t1\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tsentence1\t0\n')
        train_file.write('2\tsentence2\t1\n')
        train_file.write('3\tsentence3\t0\n')
        train_file.write('4\tsentence4\t1\n')
        train_file.write('5\tsentence5\t0\n')
        train_file.write('6\tsentence6\t0\n')


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


def is_solver_with_background(cls):
    # pylint: disable=multiple-statements
    if is_memory_network_solver(cls): return True
    if cls == MultipleChoiceSimilaritySolver: return True
    return False
