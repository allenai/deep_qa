"""
# This code implements the word overlap baseline for retreiving a table row from TableStore.
# In gets as input a question and retrieves top k high scored table rows.
# Table rows are treated as sentences for this version
# There are two options for word overlap: tf-idf and plain word matching,
# choice is identified by tf_idf flag
# The current version picks up a random sample of size "num_qa_pairs from the AristoQuiz dataset,
# runs the word_overlap and produces the following output:
# question, retrieved sentence, answer, score.  score is set to 1
# if the answer phrase is included in the retrieved sentence irrespective of context
"""

import argparse
import json
import csv
from collections import defaultdict
import operator
import math
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer as NltkPorterStemmer

np.random.seed(seed=20)
stemmer = NltkPorterStemmer()  # pylint: disable=invalid-name
# stemmer for lematizing the words
def pre_process(row):
    """
    gets a row, makes a complete sentence string, lower cases it,
    tokenizes, removes stop words and does stemming
    Returns the clean list
    -------

    """
    sentence = ' '.join(word.lower() for word in row)
    sentence_tokenized = [w for w in word_tokenize(sentence)]
    clean_row = [w for w in sentence_tokenized if w not in stopwords.words('english')]
    clean_words = [stemmer.stem(w) for w in clean_row]
    return clean_words

def get_inverse_index_dict(tables_dataset):
    """

    Parameters
    ----------
    tables_dataset

    Returns  given the table dataset this function makes table vocabulary
    and inv_index for word in table vocabulary
    note that stop words are removed and we consider stemmed version of the word
    -------

    """
    inv_index = defaultdict(set)
    tables_vocab = set()
    num_all_rows = 0
    with open(tables_dataset) as data_file:
        data = json.load(data_file)
    # make the inv_index dictionaries for all non-stop words
    for j in range(len(data["tables"])):
        table_rows = data["tables"][j]["data"] # each table row is a list, not an array
        num_all_rows = num_all_rows + len(table_rows)
        for i, table_row in enumerate(table_rows):
            words = pre_process(table_row)
            for word in words:
                tables_vocab.add(word)
                index = (j, i)
                inv_index[word].add(index)
    return inv_index, tables_vocab, num_all_rows

def get_question_answer_pairs(filename, number):
    """

    Parameters
    ----------
    filename: QA dataset
    number: the number of samples to take from it

    Returns get (question, answer) random samples from the filename,
    the number of samples is specified by "number"
    -------
    """

    questions = []
    answers = []
    with open(filename) as f_file:
        reader = csv.reader(f_file)
        for row in reader:
            questions.append(row[0])
            answers.append(row[2])
    samples_num = np.random.choice(len(questions), size=number, replace=False)
    picked_questions = [questions[i] for i in samples_num]
    picked_answers = [answers[i] for i in samples_num]
    return picked_questions, picked_answers


def score_table_row_tfidf(question, tables_dataset, table_rows_to_score,
                          inv_index, tables_vocab, num_all_rows):
    """
    Parameters
    ----------
    question
    tables_dataset
    table_rows_to_score
    inv_index
    tables_vocab
    num_all_rows

    Returns the table rows that have word overlap with question along with a tf-idf score
    -------

    """
    scored_rows = dict()
    with open(tables_dataset) as data_file:
        data = json.load(data_file)
    for (j, i) in table_rows_to_score:
        table_row = data["tables"][j]["data"][i]
        score = 0
        words_to_consider = pre_process(table_row)
        question_words = pre_process(question)
        for word in words_to_consider:
            for question_word in question_words:
                if question_word in tables_vocab and question_word == word:    # similarity measure
                    score = score +  words_to_consider.count(word) / \
                            len(words_to_consider)*math.log(num_all_rows/len(inv_index[word]))
        index = (j, i)
        scored_rows[index] = score
    return scored_rows

def score_table_row_plain(question, tables_dataset, table_rows_to_score, tables_vocab):
    """

    Parameters
    ----------
    question
    tables_dataset
    table_rows_to_score
    tables_vocab

    Returns the table rows that have word overlap with question along with a tf-idf score
    -------

    """
    scored_rows = dict()
    with open(tables_dataset) as data_file:
        data = json.load(data_file)
    for (j, i) in table_rows_to_score:
        table_row = data["tables"][j]["data"][i]
        score = 0
        words_to_consider = pre_process(table_row)
        question_words = pre_process(question)
        for word in words_to_consider:
            for question_word in question_words:
                if question_word == word and question_word in tables_vocab: # similarity measure
                    score = score + 1
        index = (j, i)
        scored_rows[index] = score
    return scored_rows

def write_top_queries(question, answer, tables_dataset, scored_rows, top_n, output_file):
    """

    Parameters
    ----------
    question
    answer
    tables_dataset
    scored_rows
    top_n
    output_file

    Returns: given the question and scored rows, it writes top scored rows in a file
    along with a label which says if the answer exists in that row.
    -------

    """
    question_score = 0
    with open(tables_dataset) as data_file:
        data = json.load(data_file)
    # sort the scored rows, this is how you sort a dictionary based on values,
    # change 1 to 0 to sort by keys
    sorted_scored_rows = sorted(scored_rows.items(), key=operator.itemgetter(1), reverse=True)
    # get top_n from the sorted dictionary
    output_list = []
    for index in range(min(top_n, len(sorted_scored_rows))):  # for rare words
        label = 0
        ((j, i), score) = sorted_scored_rows[index]
        row = ' '.join(data["tables"][j]["data"][i])
        row = ' '.join([stemmer.stem(w.lower()) for w in word_tokenize(row)])
        answer = ' '.join([stemmer.stem(w.lower()) for w in word_tokenize(answer)])
        if answer in row:
            label = 1
            question_score = question_score + 1
        retrieved_pair = (question, row, answer, score, label)
        output_list.append(retrieved_pair)
    with open(output_file, 'a') as f_file:
        writer = csv.writer(f_file)
        for pair in output_list:
            writer.writerow(pair)

def get_top_queries(question, answer, tables_dataset,
                    inv_index, tables_vocab, top_n, output_file, num_all_rows, tf_idf_flag):
    """

    Parameters
    ----------
    question
    answer
    tables_dataset
    inv_index
    tables_vocab
    top_n
    output_file
    num_all_rows
    tf_idf_flag

    Returns: get a subset of table rows that includes at least one of the words in the question,
    then send it to the appropriate scoring fn to find the topk and write it in the output
    -------

    """
    #### Do this for each question, answer pair ####

    # get a subset of table rows that includes at least one of the words in the question
    table_rows_to_score = set()
    question_words = pre_process(question)
    for question_word in question_words:
        if question_word in tables_vocab:
            table_rows_to_score = table_rows_to_score.union(inv_index[question_word])
    # given the (table_id, row_number), score the table row against the question
    if tf_idf_flag:
        scored_rows = score_table_row_tfidf(question, tables_dataset,
                                            table_rows_to_score, inv_index, tables_vocab,
                                            num_all_rows)
    else:
        scored_rows = score_table_row_plain(question, tables_dataset,
                                            table_rows_to_score, tables_vocab)
    # get top_n queries and write them to output_file
    write_top_queries(question, answer, tables_dataset, scored_rows, top_n, output_file)





def main():
    """

    Returns given a question and table store returns topk related rows
    scored by tf-idf or simple word overlap
    -------

    """
    argparser = argparse.ArgumentParser(description="Prepare ")
    argparser.add_argument("--QA_dataset", type=str, help="AristoQuiz dataset, "
                                                          "or any list of (question, answer)",
                           default="data/AristoQuizTableAlignmentV0.1.csv")
    argparser.add_argument("--tables_dataset", type=str,
                           help="the file that includes Aristo TableStore in json format. "
                                "the implementation depends on this format",
                           default='data/Grade4RemovedTablesAndRows-v2.json')
    argparser.add_argument("--output_file", type=str,
                           help="output file with question, top k retrieved table rows "
                                "one at a line,answer and score",
                           default='data/output.csv')
    argparser.add_argument("--topk", type=str, help="number of top k retrieved rows",
                           default=100)
    argparser.add_argument("--num_qa_pairs", type=str, help="number of random samples "
                                                            "taken from QA_dataset for evaluation",
                           default=1000)
    argparser.add_argument("--tf_idf_flag", type=str, help="TFIdf flag, 0 = plain word overlap, "
                                                           "1: Tf-IDF scoring of word overlap",
                           default=1)
    args = argparser.parse_args()

    inv_index, tables_vocab, num_all_rows = get_inverse_index_dict(args.tables_dataset)
    questions, answers = get_question_answer_pairs(args.QA_dataset, args.num_qa_pairs)
    for question_index in range(args.num_qa_pairs):
        get_top_queries(questions[question_index], answers[question_index], args.tables_dataset,
                        inv_index, tables_vocab, args.topk,
                        args.output_file, num_all_rows, args.tf_idf_flag)


if __name__ == '__main__':
    main()

# vim: et sw=4 sts=4
