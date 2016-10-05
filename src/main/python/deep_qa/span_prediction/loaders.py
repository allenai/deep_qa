""" This script provides data Loader functionality
"""

import numpy as np

from nltk import pos_tag
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree

import re
import json

np.random.seed(42)

STOP_WORDS = set(stopwords.words('english'))
DATA_PATH = "/Users/johannesw/Desktop/Johannes/dataset_creation/data/"
#BUSC_PATH = "/Users/johannesw/Desktop/Johannes/dataset_creation/data/part-00000"


def get_science_terms(tokens = True):
    """ loads collection of science terms.
    input: tokens (bool). If True, a set of unique individual tokens will be
        returned. If False, multi-word expressions will be returned.
    """
    with open(DATA_PATH + 'science-terms-list.txt', 'r') as openfile:
        science_terms = [line.rstrip() for line in openfile.readlines() if not '#' in line]
    if tokens:
        science_tokens = set()
        for term in science_terms:
            for token in wordpunct_tokenize(term):
                science_tokens.add(token)
        # filter out stop words.
        return science_tokens.difference(STOP_WORDS)
    return science_terms


def read_barrons():
    """ loads some barron's examples.
    """
    with open(DATA_PATH+'Barrons-4thGrade.sentences-d1/Barrons-1.sentences.txt', 'r') as readfile:
        raw_examples = [x.strip('\n') for x in readfile.readlines()]
    tokenised_examples = [wordpunct_tokenize(xpl) for xpl in raw_examples]

    # check that final token is '.'
    filtered_examples = [xpl for xpl in tokenised_examples if xpl[-1] == '.' ]
    return filtered_examples


def load_questions():
    """
     loads Omnibus-Gr04 and Omnibus-Gr08 fill-the-gap questions and provides
     them in form of statements, together with the and span position (index pair).
     """
    question_file_names = ["Omnibus-Gr08-NDMC-Train1-v1.tsv",
                        "Omnibus-Gr08-NDMC-Train2-v1.tsv",
                        "Omnibus-Gr08-NDMC-Train3-v1.tsv",
                        "Omnibus-Gr04-NDMC-Train-v1.tsv"]

    question_collection, answer_collection = [], []
    for qfn in question_file_names:
        with open(DATA_PATH+qfn, 'r') as openfile:
            for line in openfile.readlines():
                question_bit = line.split('\t')[-2]
                if not '____' in question_bit:
                    continue
                sentences = question_bit.split('.')
                if len(sentences) > 2:
                    continue
                question_collection.append(question_bit)
                answer_collection.append( line.split('\t')[3] )

    # get statements and spans.
    statements, spans = [], []
    for q,a in zip(question_collection, answer_collection):
        question_part, answers_part = q.split('.')
        answer_candidates = [a[3:] for a in answers_part.split('(')[1:]]
        Answer_Dict = dict()
        Answer_Dict['A'] = answer_candidates[0]
        Answer_Dict['B'] = answer_candidates[1]
        Answer_Dict['C'] = answer_candidates[2]
        Answer_Dict['D'] = answer_candidates[3]

        answer_tokens = wordpunct_tokenize(Answer_Dict[a])
        question_part = re.sub(r'_{2,}', ' XXXXX', question_part)
        question_tokens = wordpunct_tokenize(question_part) + ['.']  #was removed.
        position = question_tokens.index('XXXXX')

        span = (position, position+len(answer_tokens) )
        statement = question_tokens[:position] + answer_tokens + question_tokens[position+1:]
        if 'XXXXX' in ' '.join(statement): # multiple blanks in the question
            continue
        statements.append(statement)
        spans.append(span)
    return statements, spans


def write_sentences(statements):
    # writing out sentences line by line, ready for CoreNLP parser.
    with open('sentences_barrons.txt', 'w') as outfile:
        for s in statements:
            outfile.write(" ".join(s)+'\n')


def load_parses(barrons=False):
    # loads precomputed CoreNLP parse trees.
    filename = 'sentences.txt.json'
    if barrons:
        filename='sentences_barrons.txt.json'
    try:
        with open(filename, 'r') as infile:
            unsorted_parses = json.load(infile)['sentences']
    except Exception:
        print("#####   Must run CoreNLP parser first.  #####")
        return
    #json object confounds order.
    Dict = {instance['index']: instance['parse'] for instance in unsorted_parses}
    sorted_parses = [ Dict[index] for index in range(0,len(Dict)) ]

    # read out trees, into nltk.Tree format.
    trees = []
    for sentence_parse in sorted_parses:
        tree = Tree.fromstring(" ".join(sentence_parse.splitlines()))
        trees.append(tree)
    return trees


# get POS tag list
def POS_SET(statements):
    ALL_POS = set()
    n_steps = 0
    for statement in statements:
        for _, tag in pos_tag(statement):
            n_steps += 1
            ALL_POS.add(tag)
    ALL_POS = list(sorted(ALL_POS))
    return ALL_POS
