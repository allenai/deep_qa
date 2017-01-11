# -*- coding: utf-8 -*-
"""
This script takes as input CSV files from the Maluuba NewsQA dataset.
The dataset is quite dirty by default, so this script does some preprocessing
and extracts the relevant information we neeed in the deep_qa library.
"""
import json
import logging
import os
import re

from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
from scipy.stats import mode

logger = logging.getLogger(__name__) # pylint: disable=invalid-name


def main():
    LOG_FMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)
    parser = ArgumentParser(description=("Clean up a CSV file in "
                                         "the NewsQA dataset."))
    parser.add_argument('input_csv', nargs='+',
                        metavar="<input_csv>", type=str,
                        help=("Path to CSV files to clean up. Pass in "
                              "as many as you want, and the output "
                              "will be written to <input_csv>.clean"))

    arguments = parser.parse_args()
    for newsqa_file in arguments.input_csv:
        clean_newsqa_csv(newsqa_file)


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def clean_newsqa_csv(newsqa_file_path):
    logger.info("cleaning up %s", newsqa_file_path)
    # open the file as a csv
    dataframe = pd.read_csv(newsqa_file_path, encoding='utf-8')
    dirty_rows = dataframe.values.tolist()
    clean_rows = []
    clean_headers = ["question_text", "label", "answer_string", "passage"]
    for row in tqdm(dirty_rows):
        clean_row = []
        # clean up dirty file
        candidate_answers = re.split(r"\||,", row[2])
        answer_absent_prob = float(row[3]) if isfloat(row[3]) else 1.0
        passage_bad_prob = float(row[4]) if isfloat(row[4]) else 1.0
        validated_answers = row[5]
        raw_passage_text = row[6]

        # figure out the label span (answer_span)
        if validated_answers and not pd.isnull(validated_answers):
            # pick the validated answer with the most votes
            # in case of tie, pick the longest one
            validated_answers_dict = json.loads(validated_answers)
            answer_span = max(validated_answers_dict,
                              key=validated_answers_dict.get)
        else:
            # fall back and pick the candidate answer that
            # occurs most frequently.
            answer_span = mode(candidate_answers)[0][0]

        if (answer_span.lower() == "none" or answer_span.lower() == "bad_question" or
                    answer_absent_prob >= 0.5 or passage_bad_prob >= 0.5):
            continue
        initial_span_start, initial_span_end = [int(x) for x in
                                                answer_span.split(":")]
        if not raw_passage_text[initial_span_start:initial_span_end][-1].isalnum():
            initial_span_end -= 1

        raw_answer_snippet = raw_passage_text[:initial_span_start]

        # count the number of spaces to add before the answer (newlines following non-newline)
        num_spaces_added = len(re.findall("(?<=[^\\n|\\r])(\\n|\\r)",
                                          raw_answer_snippet))
        # count the number of newlines that we're going to remove
        # before the answer (all newlines before the answer)
        num_newlines_removed = len(re.findall("(\\r|\\n)", raw_answer_snippet))
        # offset refers to how much to shift the span by
        offset = (num_newlines_removed) - num_spaces_added
        # remove consecutive newlines with spaces in the raw passage text
        # to get a clean version with no linebreaks
        processed_passage_text = re.sub("(\\r|\\n)+", " ", raw_passage_text)
        # calculate the new span indices by subtracting the previously calcuated offset
        final_span_start = initial_span_start - offset
        final_span_end = initial_span_end - offset
        # build the new row of the dataset
        # question text
        clean_row.append(row[1])
        # label
        clean_row.append(str(final_span_start) + ":" + str(final_span_end))
        # answer as a string
        clean_row.append(processed_passage_text[final_span_start:final_span_end])
        # passage text
        clean_row.append(processed_passage_text)
        clean_rows.append(clean_row)
    # turn the list of rows into a dataframe, and write to CSV
    dataframe = pd.DataFrame(clean_rows, columns=clean_headers)
    folder, filename = os.path.split(newsqa_file_path)
    outdirectory = folder + "/cleaned/"
    os.makedirs(outdirectory, exist_ok=True)
    outpath = outdirectory + filename + ".clean"
    logger.info("Saving cleaned file to %s", outpath)
    dataframe.to_csv(outpath, encoding="utf-8", index=False)

if __name__ == '__main__':
    main()

