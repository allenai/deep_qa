import logging
import itertools
import os
import sys

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
import numpy
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member
# pylint: disable=wrong-import-position

import pyhocon

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.common.params import replace_none
from deep_qa.common import util
from deep_qa.contrib.background_search.vector_based_retrieval import VectorBasedRetrieval


def get_background_for_questions(retrieval: VectorBasedRetrieval,
                                 question_file: str,
                                 question_format: str,
                                 num_neighbors: int,
                                 output_file: str):
    sentences = []
    indices = []
    for line in open(question_file):
        parts = line.strip().split('\t')
        indices.append(parts[0])
        sentences.append(parts[1:])
    if question_format == 'sentence':
        # Each tuple in sentences is (sentence, label)
        queries = [sentence[0] for sentence in sentences]
    elif question_format == 'question and answer':
        # Each tuple in sentences is (sentence, options, label)
        options = [sentence[1].split("###") for sentence in sentences]
        max_num_options = max(len(o) for o in options)
        options = [pad_list_to_length(o, max_num_options) for o in options]
        # We'll make a matrix of queries here, then flatten it, and reshape it back later.  This
        # will let the encoder and the nearest neighbor algorithms be the most efficient with
        # batching, if they do anything special.
        query_matrix = [[sentence] + o for (sentence, o) in zip(sentences, options)]
        queries = list(itertools.chain.from_iterable(query_matrix))
    else:
        raise RuntimeError("Unrecognized question format: " + question_format)
    nearest_neighbors = retrieval.get_nearest_neighbors(queries, num_neighbors)
    if question_format == 'question and answer':
        # We need to do some post-processing here to sort the results from all the queries.  Note
        # that we are comparing the similarity scores from different queries here. This is not a
        # correct comparison, but we just want to push the not-so-relevant results towards the end
        # so that they can later be pruned if needed.
        num_queries_per_question = max_num_options + 1
        actual_num_neighbors = num_queries_per_question * num_neighbors

        # We flatten the nearest neighbors list, then re-group so we have something of shape
        # (num_questions, num_results_per_question).
        flattened_nearest_neighbors = list(itertools.chain.from_iterable(nearest_neighbors))
        results_by_question = util.group_by_count(flattened_nearest_neighbors,
                                                  actual_num_neighbors, (-1, -1))
        nearest_neighbors = []
        for question_results in results_by_question:
            # Sorting by similarity scores
            question_results.sort(key=lambda x: x[1])
            filtered_neighbor_indices = []
            seen_neighbors = set([])
            for passage, score in question_results:
                if len(filtered_neighbor_indices) >= num_neighbors:
                    break
                if passage not in seen_neighbors:
                    filtered_neighbor_indices.append((passage, score))
                    seen_neighbors.add(passage)
            nearest_neighbors.append(filtered_neighbor_indices)
    with open(output_file, "w") as outfile:
        for i, neighbors_with_scores in zip(indices, nearest_neighbors):
            neighbors = [n[0] for n in neighbors_with_scores]
            print("%s\t%s" % (i, "\t".join(neighbors)), file=outfile)
        outfile.close()


def pad_list_to_length(item_list, desired_length):
    if len(item_list) == desired_length:
        return item_list
    if len(item_list) > desired_length:
        raise RuntimeError("Can't pad list; it's already longer than you want")
    return item_list + [''] * (desired_length - len(item_list))


def main():
    if len(sys.argv) != 2:
        print('USAGE: retrieve_background.py [param_file]')
        sys.exit(-1)

    param_file = sys.argv[1]
    params = pyhocon.ConfigFactory.parse_file(param_file)
    params = replace_none(params)

    retrieval_params = params.pop('retrieval')
    corpus_file = params.pop('corpus', None)
    question_params = params.pop('questions')
    question_file = question_params.pop('file')
    question_format = question_params.pop('format', 'sentence')
    num_neighbors = params.pop('num_neighbors', 50)
    output_file = params.pop('output', None)
    if output_file is None:
        output_file = question_file.rsplit('.', 1)[0] + ".retrieved_background.tsv"

    retrieval = VectorBasedRetrieval(retrieval_params)
    if corpus_file is not None:
        retrieval.read_background(corpus_file)
        retrieval.fit()
        retrieval.save_model()
    else:
        retrieval.load_model()


    get_background_for_questions(retrieval, question_file, question_format, num_neighbors, output_file)

if __name__ == '__main__':
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
