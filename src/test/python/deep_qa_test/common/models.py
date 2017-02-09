# pylint: disable=invalid-name
import codecs
import gzip
import shutil

from deep_qa.models.memory_networks.memory_network import MemoryNetwork
from deep_qa.models.multiple_choice_qa.multiple_true_false_similarity import MultipleTrueFalseSimilarity

from .constants import TEST_DIR
from .constants import TRAIN_FILE
from .constants import TRAIN_BACKGROUND
from .constants import VALIDATION_FILE
from .constants import VALIDATION_BACKGROUND
from .constants import SNLI_FILE
from .constants import PRETRAINED_VECTORS_FILE
from .constants import PRETRAINED_VECTORS_GZIP


def get_model(cls, additional_arguments=None):
    params = {}
    params['save_models'] = False
    params['model_serialization_prefix'] = TEST_DIR
    params['train_files'] = [TRAIN_FILE]
    params['validation_files'] = [VALIDATION_FILE]
    params['embedding_size'] = 6
    params['encoder'] = {"default": {'type': 'bow'}}
    params['num_epochs'] = 1
    params['keras_validation_split'] = 0.0
    if is_model_with_background(cls):
        params['train_files'].append(TRAIN_BACKGROUND)
        params['validation_files'].append(VALIDATION_BACKGROUND)
    if is_memory_network(cls):
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


def write_true_false_model_files():
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
        train_file.write('2\tsentence2 word2 word3\t1\n')
        train_file.write('3\tsentence3 word2\t0\n')
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


def write_multiple_true_false_memory_network_files():
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
        train_file.write('1\ta b e i d\tanswer 1###answer2\t0\n')
        train_file.write('2\ta b c d\tanswer3###answer4\t1\n')
        train_file.write('3\te d w f d s a\tanswer5###answer6###answer9\t2\n')
        train_file.write('4\te fj k w q\tanswer7###answer8\t0\n')
    with codecs.open(TRAIN_BACKGROUND, 'w', 'utf-8') as train_background:
        train_background.write('1\tsb1\tsb2\n')
        train_background.write('2\tsb3\n')
        train_background.write('3\tsb4\n')
        train_background.write('4\tsb5\tsb6\n')


def write_who_did_what_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tpassage1\tquestion1\tanswer1###answer2\t0\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        # document, question, answers
        train_file.write('1\te q e q ka q o\ta b XXX e i d\tanswer1 word2###answer2\t0\n')
        train_file.write('2\tm a os e z p\ta b XXX c d\tanswer3###answer4\t1\n')
        train_file.write('3\tx e q m\te d w f d XXX s a\tanswer5###answer6###answer9\t2\n')
        train_file.write('4\tj aq ei q l\te fj XXX k w q\tanswer7###answer8\t0\n')


def write_span_prediction_files():
    with codecs.open(VALIDATION_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tquestion 1\tpassage with answer\t13,18\n')
    with codecs.open(TRAIN_FILE, 'w', 'utf-8') as train_file:
        train_file.write('1\tquestion 1\tpassage1 with answer1\t14,20\n')
        train_file.write('2\tquestion 2\tpassage2 with answer2\t0,8\n')
        train_file.write('3\tquestion 3\tpassage3 with answer3\t9,13\n')
        train_file.write('4\tquestion 4\tpassage4 with answer4\t14,20\n')


def write_pretrained_vector_files():
    # write the file
    with codecs.open(PRETRAINED_VECTORS_FILE, 'w', 'utf-8') as vector_file:
        vector_file.write('word2 0.21 0.57 0.51 0.31\n')
        vector_file.write('sentence1 0.81 0.48 0.19 0.47\n')
    # compress the file
    with open(PRETRAINED_VECTORS_FILE, 'rb') as f_in, gzip.open(PRETRAINED_VECTORS_GZIP, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

def is_memory_network(cls):
    if issubclass(cls, MemoryNetwork):
        return True
    return False


def is_model_with_background(cls):
    # pylint: disable=multiple-statements
    if is_memory_network(cls): return True
    if cls == MultipleTrueFalseSimilarity: return True
    return False
