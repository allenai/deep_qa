""" This script provides data Loader functionality
"""
import re
import json
from typing import List

import numpy as np

from nltk import pos_tag
from nltk.tokenize import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.tree import Tree


np.random.seed(42)
STOP_WORDS = set(stopwords.words('english'))


def get_science_terms(data_path, split_into_tokens: bool=True):
    """ loads collection of science terms.
    input: tokens (bool). If True, a set of unique individual tokens will be
        returned. If False, multi-word expressions will be returned.
    """
    with open(data_path + 'science-terms-list.txt', 'r') as openfile:
        science_terms = [line.rstrip() for line in openfile.readlines() if '#' not in line]
    if split_into_tokens:
        science_tokens = set()
        for term in science_terms:
            for token in wordpunct_tokenize(term):
                science_tokens.add(token)
        # filter out stop words.
        return science_tokens.difference(STOP_WORDS)
    return science_terms


def read_barrons(filename):
    """ loads some barron's examples.
    """
    with open(filename, 'r') as readfile:
        raw_examples = [x.strip('\n') for x in readfile.readlines()]
    tokenised_examples = [wordpunct_tokenize(x) for x in raw_examples]

    # Not all sentences end with full stop, this can mess up with the CoreNLP parser.
    # filter out all cases that don't end with full stop.
    filtered_examples = [x for x in tokenised_examples if x[-1] == '.']
    return filtered_examples


def load_questions(data_path):
    """
     loads Omnibus-Gr04 and Omnibus-Gr08 fill-the-gap questions and provides
     them in form of statements, together with the span position (index pair).
     We only keep questions with a gap, e.g. "Elephants live in _____.".
     Cases with more than one gap are filtered out.
     Multi-sentence cases are filtered out.
     Returns: List of tuples,
        - first entry: List of List of strings
        - second entry: pair of integers
     """
    question_file_names = ["Omnibus-Gr08-NDMC-Train1-v1.tsv",
                           "Omnibus-Gr08-NDMC-Train2-v1.tsv",
                           "Omnibus-Gr08-NDMC-Train3-v1.tsv",
                           "Omnibus-Gr04-NDMC-Train-v1.tsv"]

    questions = []
    for file_name in question_file_names:
        with open(data_path+file_name, 'r') as openfile:
            for line in openfile.readlines():
                # second to last column in tsv contains question statement
                question_bit = line.split('\t')[-2]
                if '____' not in question_bit:
                    continue
                if len(question_bit.split('.')) > 2:
                    continue
                # column in the tsv file contains the correct answer
                correct_answer = line.split('\t')[3]
                questions.append((question_bit, correct_answer))

    # List with (statement, span) tuples.
    statements_with_gaps = []
    for statement, correct_answer in questions:
        question_part, answers_part = statement.split('.')
        answer_candidates = [a[3:] for a in answers_part.split('(')[1:]]
        answer_dict = dict()
        answer_dict['A'] = answer_candidates[0]
        answer_dict['B'] = answer_candidates[1]
        answer_dict['C'] = answer_candidates[2]
        answer_dict['D'] = answer_candidates[3]

        answer_tokens = wordpunct_tokenize(answer_dict[correct_answer])
        # the number of underscores _ was not consistent across the questions.
        # So they are replaced with a consistent placeholder token 'XXXXX'.
        question_part = re.sub(r'_{2,}', ' XXXXX', question_part)
        question_tokens = wordpunct_tokenize(question_part) + ['.']  #was removed.
        position = question_tokens.index('XXXXX')

        span = (position, position+len(answer_tokens))
        statement = question_tokens[:position] + answer_tokens + question_tokens[position+1:]
        if 'XXXXX' in ' '.join(statement): # multiple blanks in the question
            continue
        statements_with_gaps.append((statement, span))
    return statements_with_gaps


def write_sentences(statements: List[List[str]], filename='sentences.txt'):
    # writing out sentences line by line, ready for CoreNLP parser.
    with open(filename, 'w') as outfile:
        for statement in statements:
            outfile.write(" ".join(statement)+'\n')


def load_parses(filename='sentences.txt.json'):
    # loads precomputed CoreNLP parse trees.
    try:
        with open(filename, 'r') as infile:
            parses = json.load(infile)['sentences']
            #parses.sort(key=lambda x: x['index'])
            parse_dict = {instance['index']: instance['parse'] for instance in parses}
            parses = [parse_dict[index] for index in range(0, len(parse_dict))]
    except FileNotFoundError:
        print("#####   Must run CoreNLP parser first.  #####")
        quit()

    # read out trees, into nltk.Tree format.
    trees = []
    for sentence_parse in parses:
        tree = Tree.fromstring(" ".join(sentence_parse.splitlines()))
        trees.append(tree)
    return trees


# get POS tag list
def get_pos_set(statements: List[List[str]]):
    all_pos = set()
    n_steps = 0
    for statement in statements:
        for _, tag in pos_tag(statement):
            n_steps += 1
            all_pos.add(tag)
    all_pos = sorted(all_pos)
    return all_pos
