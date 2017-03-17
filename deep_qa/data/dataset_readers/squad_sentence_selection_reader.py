import argparse
import json
import logging
import os
import random

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
                        help=('Output directory. Make sure to end the string with a /'))
    args = parser.parse_args()
    reader = SquadSentenceSelectionReader(args.output_directory)
    reader.read_file(args.input_filename)


class SquadSentenceSelectionReader():
    def __init__(self, output_directory=None):
        self.output_directory = output_directory

    def read_file(self, input_filepath):
        # Import is here, since it isn't necessary by default.
        import nltk
        processed_rows = []
        logger.info("Reading file at %s", input_filepath)
        with open(input_filepath) as dataset_file:
            dataset_json = json.load(dataset_file)
            dataset = dataset_json['data']
        for article in tqdm(dataset):
            for paragraph in article['paragraphs']:
                context_article = paragraph["context"]
                # Split the context_article into a list of sentences.
                sentences = nltk.sent_tokenize(context_article)

                # Make a dict from span indices to sentence. The end span is
                # exclusive, and the start span is inclusive.
                span_to_sentence_index = {}
                current_index = 0
                for sentence in sentences:
                    sentence_len = len(sentence)
                    # Need to add one to the end index to account for the
                    # trailing space after punctuation that is stripped by NLTK.
                    span_to_sentence_index[(current_index,
                                            current_index + sentence_len + 1)] = sentence
                    current_index += sentence_len + 1
                for question_answer in paragraph['qas']:
                    # Shuffle the sentences.
                    random.shuffle(sentences)
                    question_text = question_answer["question"]
                    answer_start_index = question_answer["answers"][0]["answer_start"]
                    # Get the full sentence corresponding to the answer.
                    answer_sentence = None
                    for span_tuple in span_to_sentence_index:
                        start_span, end_span = span_tuple
                        if start_span <= answer_start_index and answer_start_index < end_span:
                            answer_sentence = span_to_sentence_index[span_tuple]
                            break
                    # This runs if we did not break, and thus
                    # answer_sentence is None
                    else:
                        raise ValueError("Index of answer start was out of bounds. "
                                         "This should never happen, please raise "
                                         "an issue on GitHub.")
                    # Now that we have the string of the full sentence, we need to
                    # search for it in our shuffled list to get the index.
                    answer_index = sentences.index(answer_sentence)

                    # Now we can make the string representation and add this
                    # to the list of processed_rows.
                    row_text = (question_text + "\t" + '###'.join(sentences) +
                                "\t" + str(answer_index))
                    processed_rows.append(row_text)
        input_directory, input_filename = os.path.split(input_filepath)
        output_filename = "processed_" + input_filename + ".tsv"
        if self.output_directory:
            # Use a custom output directory.
            output_filepath = os.path.join(self.output_directory, output_filename)
        else:
            # Make a subdirectory of the input_directory called "processed",
            # and write the file there
            if not os.path.exists(os.path.join(input_directory, "processed")):
                os.makedirs(os.path.join(input_directory, "processed"))
            output_filepath = os.path.join(input_directory, "processed", output_filename)
        with open(output_filepath, 'w') as file_handler:
            for row in processed_rows:
                file_handler.write("{}\n".format(row))
        logger.info("Wrote output to %s", output_filepath)
        return output_filepath

if __name__ == '__main__':
    main()
