import argparse
from collections import Counter
import json
import logging
import os
import random
from typing import List, Tuple

import numpy
from tqdm import tqdm

logger = logging.getLogger(__name__) # pylint: disable=invalid-name

random.seed(2157)


def main():
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format)

    parser = argparse.ArgumentParser(description=("Parse a SQuAD v1.1 data file for "
                                                  "use by the SentenceSelectionInstance"))
    parser.add_argument('input_filename', help='Input SQuAD json file.')
    parser.add_argument('--output_directory',
                        help='Output directory. Make sure to end the string with a /')
    parser.add_argument('--negatives', default='paragraph', help="See class docstring")
    args = parser.parse_args()
    reader = SquadSentenceSelectionReader(args.output_directory, args.negatives)
    reader.read_file(args.input_filename)


class SquadSentenceSelectionReader():
    """
    Parameters
    ----------
    output_directory: str, optional (default=None)
        If you want the output stored somewhere other than in a ``processed/`` subdirectory next to
        the input, you can override the default with this parameter.
    negative_sentence_selection: str, optional (default="paragraph")
        A comma-separated list of methods to use to generate negative sentences in the data.

        There are three options here:

        (1) "paragraph", which means to use as negative sentences all other sentences in the same
            paragraph as the correct answer sentence.
        (2) "random-[int]", which means to randomly select [int] sentences from all SQuAD sentences
            to use as negative sentences.
        (3) "pad-to-[int]", which means to randomly select sentences from all SQuAD sentences until
            there are a total of [int] sentences.  This will not remove any previously selected
            sentences if you already have more than [int].
        (4) "question", which means to use as a negative sentence the `question` itself.
        (5) "questions-random-[int]", which means to select [int] random `questions` from SQuAD to
            use as negative sentences (this could include the question corresponding to the
            example; we don't filter out that case).

        We will process these options in order, so the "pad-to-[int]" option mostly only makes
        sense as the last option.
    """
    def __init__(self, output_directory: str=None, negative_sentence_selection: str="paragraph"):
        self.output_directory = output_directory
        self.negative_sentence_selection_methods = negative_sentence_selection.split(",")

        # Initializing some data structures here that will be useful when reading a file.
        # Maps sentence strings to sentence indices
        self.sentence_to_id = {}
        # Maps sentence indices to sentence strings
        self.id_to_sentence = {}
        # Maps paragraph ids to lists of contained sentence ids
        self.paragraph_sentences = {}
        # Maps sentence ids to the containing paragraph id.
        self.sentence_paragraph_map = {}
        # Maps question strings to question indices
        self.question_to_id = {}
        # Maps question indices to question strings
        self.id_to_question = {}

    def _clear_state(self):
        self.sentence_to_id.clear()
        self.id_to_sentence.clear()
        self.paragraph_sentences.clear()
        self.sentence_paragraph_map.clear()
        self.question_to_id.clear()
        self.id_to_question.clear()

    def _get_sentence_choices(self, question_id: int, answer_id: int) -> Tuple[List[str], int]:
        # Because sentences and questions have different indices, we need this to hold tuples of
        # ("sentence", id) or ("question", id), instead of just single ids.
        negative_sentences = set()
        for selection_method in self.negative_sentence_selection_methods:
            if selection_method == 'paragraph':
                paragraph_id = self.sentence_paragraph_map[answer_id]
                paragraph_sentences = self.paragraph_sentences[paragraph_id]
                negative_sentences.update(("sentence", sentence_id)
                                          for sentence_id in paragraph_sentences
                                          if sentence_id != answer_id)
            elif selection_method.startswith("random-"):
                num_to_pick = int(selection_method.split('-')[1])
                num_sentences = len(self.sentence_to_id)
                # We'll ignore here the small probability that we pick `answer_id`, or a
                # sentence we've chosen previously.
                selected_ids = numpy.random.choice(num_sentences, (num_to_pick,), replace=False)
                negative_sentences.update(("sentence", sentence_id)
                                          for sentence_id in selected_ids
                                          if sentence_id != answer_id)
            elif selection_method.startswith("pad-to-"):
                desired_num_sentences = int(selection_method.split('-')[2])
                # Because we want to pad to a specific number of sentences, we'll do the choice
                # logic in a loop, to be sure we actually get to the right number.
                while desired_num_sentences > len(negative_sentences):
                    num_to_pick = desired_num_sentences - len(negative_sentences)
                    num_sentences = len(self.sentence_to_id)
                    if num_to_pick > num_sentences:
                        raise RuntimeError("Not enough sentences to pick from")
                    selected_ids = numpy.random.choice(num_sentences, (num_to_pick,), replace=False)
                    negative_sentences.update(("sentence", sentence_id)
                                              for sentence_id in selected_ids
                                              if sentence_id != answer_id)
            elif selection_method == "question":
                negative_sentences.add(("question", question_id))
            elif selection_method.startswith("questions-random-"):
                num_to_pick = int(selection_method.split('-')[2])
                num_questions = len(self.question_to_id)
                # We'll ignore here the small probability that we pick `question_id`, or a
                # question we've chosen previously.
                selected_ids = numpy.random.choice(num_questions, (num_to_pick,), replace=False)
                negative_sentences.update(("question", q_id) for q_id in selected_ids)
            else:
                raise RuntimeError("Unrecognized selection method:", selection_method)
        choices = list(negative_sentences) + [("sentence", answer_id)]
        random.shuffle(choices)
        correct_choice = choices.index(("sentence", answer_id))
        sentence_choices = []
        for sentence_type, index in choices:
            if sentence_type == "sentence":
                sentence_choices.append(self.id_to_sentence[index])
            else:
                sentence_choices.append(self.id_to_question[index])
        return sentence_choices, correct_choice

    def read_file(self, input_filepath: str):
        # Import is here, since it isn't necessary by default.
        import nltk
        self._clear_state()

        # Holds tuples of (question_text, answer_sentence_id)
        questions = []
        logger.info("Reading file at %s", input_filepath)
        with open(input_filepath) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        logger.info("Reading the dataset")
        for article in tqdm(dataset):
            for paragraph in article['paragraphs']:
                paragraph_id = len(self.paragraph_sentences)
                self.paragraph_sentences[paragraph_id] = []

                context_article = paragraph["context"]
                # replace newlines in the context article
                cleaned_context_article = context_article.replace("\n", "")

                # Split the cleaned_context_article into a list of sentences.
                sentences = nltk.sent_tokenize(cleaned_context_article)

                # Make a dict from span indices to sentence. The end span is
                # exclusive, and the start span is inclusive.
                span_to_sentence_index = {}
                current_index = 0
                for sentence in sentences:
                    sentence_id = len(self.sentence_to_id)
                    self.sentence_to_id[sentence] = sentence_id
                    self.id_to_sentence[sentence_id] = sentence
                    self.sentence_paragraph_map[sentence_id] = paragraph_id
                    self.paragraph_sentences[paragraph_id].append(sentence_id)

                    sentence_len = len(sentence)
                    # Need to add one to the end index to account for the
                    # trailing space after punctuation that is stripped by NLTK.
                    span_to_sentence_index[(current_index,
                                            current_index + sentence_len + 1)] = sentence
                    current_index += sentence_len + 1
                for question_answer in paragraph['qas']:
                    question_text = question_answer["question"].strip()
                    question_id = len(self.question_to_id)
                    self.question_to_id[question_text] = question_id
                    self.id_to_question[question_id] = question_text

                    # There may be multiple answer annotations, so pick the one
                    # that occurs the most.
                    candidate_answer_start_indices = Counter()
                    for answer in question_answer["answers"]:
                        candidate_answer_start_indices[answer["answer_start"]] += 1
                    answer_start_index, _ = candidate_answer_start_indices.most_common(1)[0]

                    # Get the full sentence corresponding to the answer.
                    answer_sentence = None
                    for span_tuple in span_to_sentence_index:
                        start_span, end_span = span_tuple
                        if start_span <= answer_start_index and answer_start_index < end_span:
                            answer_sentence = span_to_sentence_index[span_tuple]
                            break
                    else:  # no break
                        raise ValueError("Index of answer start was out of bounds. "
                                         "This should never happen, please raise "
                                         "an issue on GitHub.")

                    # Now that we have the string of the full sentence, we need to
                    # search for it in our shuffled list to get the index.
                    answer_id = self.sentence_to_id[answer_sentence]

                    # Now we can make the string representation and add this
                    # to the list of processed_rows.
                    questions.append((question_id, answer_id))
        processed_rows = []
        logger.info("Processing questions into training instances")
        for question_id, answer_id in tqdm(questions):
            sentence_choices, correct_choice = self._get_sentence_choices(question_id, answer_id)
            question_text = self.id_to_question[question_id]
            row_text = (question_text + "\t" + '###'.join(sentence_choices) +
                        "\t" + str(correct_choice))
            processed_rows.append(row_text)

        logger.info("Writing output file")
        input_directory, input_filename = os.path.split(input_filepath)
        output_filename = "sentence_selection_" + input_filename + ".tsv"
        if self.output_directory:
            # Use a custom output directory.
            output_filepath = os.path.join(self.output_directory, output_filename)
        else:
            # Make a subdirectory of the input_directory called "processed",
            # and write the file there
            if not os.path.exists(os.path.join(input_directory, "processed")):
                os.makedirs(os.path.join(input_directory, "processed"))
            output_filepath = os.path.join(input_directory, "processed",
                                           output_filename)
        with open(output_filepath, 'w') as file_handler:
            for row in processed_rows:
                file_handler.write("{}\n".format(row))
        logger.info("Wrote output to %s", output_filepath)
        return output_filepath

if __name__ == '__main__':
    main()
