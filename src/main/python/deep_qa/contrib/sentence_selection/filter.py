"""
This module provides functionalities related to a filter model for identifying
simple declarative science facts.
The filter model can be applied to individual sentences as well as paragraphs,
tested on annotated data, and the outputs can be written into formats amenable
to the Amazon Mechanical Turk task which uses these filter outputs.
"""

import os
import re
import csv
import random
from typing import List
from pprint import pprint
from collections import defaultdict

from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag



def sentence_splitter(long_string: str):
    '''
    Simple sentence splitting function.
    Input: long string with one or more sentences.
    Output: list of individual sentences (strings).
    '''
    pattern = r'\.(?!=*\d)'
    return [x.lstrip()+'.' for x in re.split(pattern, long_string) if x]


def regexp_delete_parentheses(expression: str):
    """
    removes parentheses and what is between them from a string.
    Input: string
    Output: string (with removed parentheses)
    """
    pattern = r' \((?<=\()[^\(]*(?=\))\)'
    return re.sub(pattern, "", expression)


class SentenceFilter():
    """
    This class provides the main sentence filtering functionality.
    Most methods are auxiliary. The core piece is the sentence_filter method,
    which filters out simple declarative statements about science from arbitrary
    statements.
    """
    def __init__(self, names_files, discourse_markers_file):
        self.names = self.load_first_names(names_files)
        self.discourse_markers = self.load_discourse_markers(discourse_markers_file)


    @staticmethod
    def load_discourse_markers(discourse_markers_file_name: str):
        """
        Loads a set of known discourse markers ('however', 'thus', ...), collected by
        myself (JW) and returns them as set.
        """
        with open(discourse_markers_file_name, 'r') as openfile:
            discourse_markers = set(line.rstrip() for line in openfile.readlines())
        return discourse_markers


    @staticmethod
    def load_first_names(names_file_name_list: List[str],
                         discard_science_names: bool=True):
        """
        Loads known first names from list of files (e.g. 'male.txt' and 'female.txt').
        The names are expected to be one per line.
        If discard_science_names is True, known names which also
        have a meaning in the context of science exam context will be discarded.
        Duplicate names will be removed.
        """

        # iterate over files, load names.
        names = set()    # no duplicates.
        for file_name in names_file_name_list:
            with open(file_name, 'r') as openfile:
                names.update(set(line.rstrip().lower() for line in openfile.readlines()\
                                 if '#' not in line))


        if discard_science_names:
            names = names.difference(["star", "rock", 'gene', "wood", 'venus', \
                                      'major', 'coral', 'violet', 'red', 'kelvin', \
                                      'june', 'diamond', 'van', "chrystal", 'way', \
                                      'gill', 'sky', 'newton', 'calvin', 'ash', \
                                      'sandy', 'noble', 'bird', 'mendel', 'moss', 'bud',\
                                      'storm', 'forest', 'april', 'sunny', 'flower', \
                                      'bay', 'jade', 'chance', 'ray', 'ambrosia', \
                                      'clay', 'goose', 'cliff', 'honey', 'kin'])
        return names


    @staticmethod
    def number_of_capitalized_tokens(expression: str):
        '''
        returns the number of capitalized tokens found in an expression (i.e. the
        number of words for which the first character is upper case).
        '''
        return sum([token.istitle() for token in wordpunct_tokenize(expression)])


    @staticmethod
    def get_special_characters(expression: str):
        """
        Returns the special characters found in an expression as set.
        """
        return {'?', '!', '$', '(', ')', '“', '&', '$', '⇒', '"', '\\', '{', '<', \
                '>', '[', ']', '*', '”', "'", '►', '→'}.intersection(expression)


    @staticmethod
    def personal_pronouns(tokens: List[str]):
        """
        Returns the personal pronouns found in a list of tokens as set.
        [not including the neutral third person cases 'they' and 'their']
        [This is intended for a sentence filter to identify science facts]
        """
        return {'i', 'my', 'me', 'you', 'your', 'yours',\
                'she', 'he', 'him', 'her', 'his', 'hers',\
                'we', 'us', 'our', 'ours',\
                'them', 'theirs'}.intersection(tokens)


    def clean_up_statement(self, statement: str):
        """
        This function takes a statement and cleans it, i.e. removes undesirable parts.
        This includes stripping it of undesirable characters and discourse markers,
        and doing some punctuation checks.
        The output is a (cleaned-up) string.
        """

        if not statement[-1] == '.':    # no full stop at the end?
            statement += '.'

        # remove some undesirable characters
        statement.strip('•')
        statement.strip('\t')
        statement.strip('"')

        tokens = wordpunct_tokenize(statement.lower())

        # some discourse markers at the sentence beginning which can be safely
        # removed without damaging the sentence
        if tokens[0] in {'still', 'so', 'or', 'and'}:
            statement = statement[len(tokens[0])+1:]

        # delete any instance of 'also' in the sentence
        statement = statement.replace(" also", "")

        # spans of up to length 4 beginning at sentences start.
        first_token_spans = [" ".join(tokens[:i]) for i in range(1, 5, 1)]

        # any discourse marker among these first 4 spans?
        discourse_markers_found = self.discourse_markers.intersection(first_token_spans)
        if discourse_markers_found:
            # identify the discourse marker
            detected_marker = [x for x in discourse_markers_found][0]
            # remove the discourse marker from the statement
            statement = statement.replace(detected_marker, "")

            # if there was a comma after the detected discourse marker, clean up
            if statement[0] == ',':
                statement = statement[1:].lstrip()
                statement = statement[0].capitalize() + statement[1:]

        return statement



    def sentence_filter(self, statements: List[str]):
        """
        This function filters a list of statements and yields those that pass the filter.
        The filter is intended to keep simple declarative statements about science,
        filtering out anything else.
        """
        # looping over list of given statements, yielding those that are accepted,
        # continuing for those that should be filtered out.
        for statement_index, statement in enumerate(statements):
            if statement_index%1000 == 0 and statement_index > 1:
                print('filtering sentences', statement_index, '/', \
                      len(statements))

            if not statement:   # empty string
                continue


            statement = self.clean_up_statement(statement)
            tokens = wordpunct_tokenize(statement.lower())


            if self.get_special_characters(statement):
                continue

            # token length is a good proxy for syntactic complexity
            if len(tokens) < 7:
                continue
            if len(tokens) > 19:
                continue

            # sentences with conjunctions and subclauses often have several commas
            if tokens.count(',') >= 3:
                continue



            pos_tags = [x[1] for x in pos_tag(tokens)]

            # statement begins with personal pronoun (>hinting at coreference), or verb?
            if pos_tags[0] in {'PRP', 'PRP$', 'VBZ'}:
                continue

            if self.personal_pronouns(tokens):   # if pos-tags didn't work, second check
                continue

            # sentence contains verb phrase? If not drop it.
            if not any([tag in ['VBP', 'VB', 'VBZ'] for tag in pos_tags]):
                continue

            if tokens[0] == 'that':
                continue

            # some specific tokens (hand-picked) that I found to be very frequent
            # among undesirable sentences.
            if {'this', 'these', 'those',
                        'please', 'should', 'shall',
                        'must', "would", "could", 'imagine',
                        'students', 'student', "teacher", "instructions", "instruction", "interview",
                        "answer", "question", "discussed", "page", "pages",
                        'table', 'tables', 'worksheet', 'shown', 'studies',
                        'never', 'always', 'everything', 'nothing',
                        'google', "download", "browser", "blog", "unavailable", "available",
                        "newspaper", 'committed', 'here', 'now', 'it' 'let', 'another',
                        'online', 'court', 'records', 'wikipedia', 'download',
                        "text", "graph", "information", "equation", "represent", "font",
                        "county", "country", "counties", "countries",
                        "consultant", "claim", "best",
                        "definitely", "certainly", "will",
                        "various", "previous",
                        "prayer", "prayers", 'although', 'more',
                        "initially", 'eventually', 'finally', 'subsequently', 'then'}\
                        .intersection(tokens):
                continue

            # some undesirable multi-word expressions
            # also: remove statements that contain hypertext references (probably markup)
            for expression in ["set in", "any more", 'examples are', 'even if', 'http']:
                if expression in statement:
                    continue

            # imperative phrases. Example -  Instructions like: "Use glue to attach pasta."
            if pos_tags.count('VB') == 1 and 'VBP' not in pos_tags:
                if pos_tags[pos_tags.index('VB') - 1] == 'TO':
                    continue

            #English imperative phrases start with the verb.
            if pos_tags[0] in {'VB', 'VBP'}:
                continue

            # specific to science context: remove unspecific species cases:
            # 'Juveniles and males remain in soil.'
            if 'males' in statement or 'females' in statement\
                and tokens[-2] not in ['males', 'females']:
                continue

            if statement[0] == ',':
                continue

            if self.number_of_capitalized_tokens(statement) >= 3:
                continue

            if statement.count('.') > 1:
                continue

            if any([':' in token for token in tokens]):
                continue

            uncased_tokens = wordpunct_tokenize(statement)
            if any([x.isupper() for x in uncased_tokens]):
                continue

            if '   ' in statement:
                continue


            # any first name in the statement? If so drop it. Don't want prose.
            names_found = self.names.intersection(tokens)
            if names_found:
                capitalized_names = set(n.capitalize() for n in names_found)
                if any([name in statement for name in capitalized_names]):
                    # see what name was found
                    #print(statement, self.names.intersection(tokens))
                    continue


            # catch unicode replacement symbols,
            # because they break the Core NLP parser (and the MTurk uploader)
            try:
                _ = statement.encode("ASCII")
            except UnicodeEncodeError:
                continue

            # some other undesirable first tokens of the statement
            if tokens[0] in {'well', 'what', 'as', 'do', 'once', 'how', 'that', \
                                'welcome', 'by', 'then', 'other', 'following', \
                                'eventually', 'to', 'step', 'even', 'then'}:
                continue

            # some other undesirable multi-word expressions at the beginning of the sentence
            first_token_spans = [" ".join(tokens[:i]) for i in range(1, 5, 1)]
            if {'whereas', 'mind you', 'if it', 'in addition', 'such', 'as well', 'note that'}\
                .intersection(first_token_spans):
                continue

            # if the statement has survived all these previous conditions, it is yielded.
            yield statement






def load_paragraphs_from_files(file_name_list: List[str], shuffled: bool=True):
    """
    This function loads paragraphs from multiple files.
    A paragraph is just a string.
    It can, but does not need to contain multiple sentences.
    The function takes a list of file names for files.
    Each of these files is intended to contain multiple paragraphs
    (e.g. a list of science textbooks) with one paragraph per line.
    The function returns a list with these paragraphs, with an option to have it
    shuffled, otherwise it's sorted.
    Duplicates are removed.
    Note: Some textbook files come in the format
            chapter index  \t   paragraph.
    In this case the chapter index is removed.
    """

    # use set to avoid duplicate paragraphs
    paragraphs_collection = set()

    for filename in file_name_list:
        if '.DS_Store' in filename:
            continue
        print("reading textbook paragraphs from file ", filename, '...')
        with open(filename, 'r') as openfile:
            # expecting one paragraph per line
            paragraphs = [x.strip() for x in openfile.readlines() if x]

            # in case there's a tab-separated chapter index at the paragraph start
            if any(['\t' in p for p in paragraphs[0]]):
                # remove the chapter index
                paragraphs = [p.split('\t')[1] for p in paragraphs]

            paragraphs_collection.update(paragraphs)

    if shuffled:
        paragraphs_collection = sorted(paragraphs_collection, key=lambda x: x)
        random.shuffle(paragraphs_collection)
    return paragraphs_collection




def filter_paragraphs(paragraphs: List[str], not_use_these_paragraphs: List[str],
                      filter_model: SentenceFilter,
                      write_file_name: str, threshold: int=0.5):
    """
    Summary function to apply the sentence filter on a list of paragraphs.
    Given a list of input paragraphs, the sentence filter is applied on each
    sentence of it. If the proportion of sentences passing the filter is larger
    than the threshold, the paragraph is accepted.
    Accepted paragraphs are written to an output file.
    Inputs:
        - paragraphs: A list of paragraphs (each a string with multiple sentences.)
        - not_use_these_paragraphs: A list of paragraphs that should not be used.
                                  This can, for example, be paragraphs known to be bad,
                                  or already used and thus undesirable.
        - write_file_name: writing outputs to file specified with this name
        - threshold: minimum proportion of sentences in a paragraph necessary to
                     have the paragraph be accepted.
    """

    filtered_paragraphs_with_ratio = []
    with open(write_file_name, 'w') as outfile:
        for paragraph_index, paragraph in enumerate(paragraphs):
            if paragraph_index % 1000 == 0 and paragraph_index > 1:
                print('filtering paragraphs...', paragraph_index, '/', len(paragraphs))

            sentences_this_paragraph = sentence_splitter(paragraph)
            if len(sentences_this_paragraph) == 0:
                continue


            if len(sentences_this_paragraph) > 20:
                continue

            # run the sentence filter over every sentence in this paragraph individually
            sentences_which_passed = len([x for x in filter_model\
                                    .sentence_filter(sentences_this_paragraph)])

            # ratio = how many are left after filtering : before
            ratio = sentences_which_passed / len(sentences_this_paragraph)

            if ratio < threshold:
                # less than the desired proportion of sentences in the paragraph pass the filter
                continue
            filtered_paragraphs_with_ratio.append((sentences_this_paragraph, ratio))

        # write the surviving paragraphs to a file.
        written_paragraphs = set()   # to skip duplicates
        for sentences_this_paragraph, _ in sorted(filtered_paragraphs_with_ratio, \
                                                  key=lambda x: x[1], reverse=True):
            paragraph_string = " ".join(sentences_this_paragraph)+'\n'

            if paragraph_string in written_paragraphs:
                continue    # Don't write duplicates.
            elif paragraph_string in not_use_these_paragraphs:
                continue    # Don't write if paragraph is in list of those that shouldn't be used.
            else:
                write_this = paragraph_string
                outfile.write(write_this)
                written_paragraphs.add(write_this)




def identify_used_paragraphs(input_file_name):
    """
    A function that is helpful when reading out the results file of a Mechanical Turk task,
    identifying the paragraphs that were used in the HIT.
    The motivation is that these should not be used any more in subsequent HITs.
    Such paragraphs will be returned in two lists.
    The first list contains the paragraphs already used by Turkers for formulating a question.
    The second list contains paragraphs that turkers couldn't make any use of at all.
    """
    # read csv file (MTurk results)
    with open(input_file_name, 'r') as infile:
        reader = csv.DictReader(infile)
        data = defaultdict(list)
        for row in reader:
            for header, value in row.items():
                data[header].append(value)


    # the answer selected by the turker
    snippets_chosen = data['Answer.Q4Answer']

    # the paragraphs proposed to the turker in this HIT
    proposed_three_snippets = [x for x in zip(data['Input.snippet1'], \
                                              data['Input.snippet2'], \
                                              data['Input.snippet3'])]

    used_up_paragraphs = []
    bad_paragraphs = []

    # looping over HITs
    for decision_among_three, paragraph_triple in zip(snippets_chosen, proposed_three_snippets):
        if decision_among_three == 'Snippet 1':
            used_up_paragraphs.append(paragraph_triple[0])
        elif decision_among_three == 'Snippet 2':
            used_up_paragraphs.append(paragraph_triple[1])
        elif decision_among_three == 'Snippet 3':
            used_up_paragraphs.append(paragraph_triple[2])
        elif decision_among_three == \
            "None of the snippets (briefly explain below why this wasn't possible!)":
            bad_paragraphs.extend(paragraph_triple)
        elif decision_among_three == '':
            continue

    return used_up_paragraphs, bad_paragraphs



def load_annotated_filtering_data(input_sentences_file_name: str,
                                  selected_sentences_file_name: str):
    """
    Function for loading two lists of sentences. Basically the gold annotations
    for how the sentence filter should work optimally.
    Given are a file name with a file for input sentences, and another file name with
    selected sentences (which must be a subset of the input sentences, and one per line),
    returns: sentences_x: List[str] of input sentences (X in the sense of 'input')
    returns: sentences_y: List[str] of selected sentences (Y in the sense of 'output')
    """

    input_content = load_paragraphs_from_files([input_sentences_file_name])
    sentences_x = []
    for line in input_content:
        sentences_x.extend(sentence_splitter(line))

    selected_content = load_paragraphs_from_files([selected_sentences_file_name])
    sentences_y = [y.strip() for y in selected_content]

    return sentences_x, sentences_y



def split_annotated_data(input_sentences: List[str], desired_output_sentences: List[str],
                         split_point: int=120):
    """
    Splits data into training and validation part.
    Task: Filter out the correct sentences from a list of input sentences.
        Input: input_sentences is such a set of input sentences
        Input: desired_output_sentences are those among the input sentences which
               should NOT be filtered out, i.e. which survive the filtering.
        Input: split_point: The index where the dataset should be split into two.
    """
    train_input = input_sentences[:input_sentences.index(desired_output_sentences[split_point])]
    train_output = desired_output_sentences[:split_point]
    valid_input = input_sentences[input_sentences.index(desired_output_sentences[split_point]):]
    valid_output = desired_output_sentences[split_point:]
    return train_input, train_output, valid_input, valid_output



def compute_filtering_metrics(y_gold: List[str], y_predicted: List[str]):
    """
    Given list of gold instances and a list of predicted instances,
    compute and print out Precision, Recall, F1 metrics.
    Additionally print out some examples of false positives and false negatives.
    Input: y_gold: A list of strings (sentences)
    Input: y_predicted: A list of strings (sentences)
    """

    y_gold = set(y_gold)
    y_predicted = set(y_predicted)
    true_positives = y_gold.intersection(y_predicted)
    false_positives = y_predicted.difference(y_gold)
    false_negatives = y_gold.difference(y_predicted)

    precision = len(true_positives) / len(y_predicted)
    recall = len(true_positives) / len(y_gold)
    f_one = 2.0*precision*recall/(precision+recall) if precision or recall else 0.0
    print("===============================================")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f_one)
    print("===============================================")
    print("Example False Negatives (filtered out, though should not be:)")
    pprint(sorted(false_negatives)[:10])
    print("===============================================")
    print("Example False Positives (not filtered out, though should be:)")
    pprint(sorted(false_positives)[:10])



def filter_annotated_data(filter_model: SentenceFilter):
    """
    This function runs the filter model over some annotated data,
    i.e. input data (x) and desirable output data (y) which a perfect filter
    would produce.
    """
    # load some annotated data
    filter_input_file_name = '/Users/johannesw/Documents/sentence_filter_input_output/X.txt'
    desired_output_file_name = '/Users/johannesw/Documents/sentence_filter_input_output/Y.txt'

    input_statements, desired_statements = load_annotated_filtering_data( \
                                filter_input_file_name, desired_output_file_name)

    # see how well the filter works on this annotated data
    train_in, train_out, valid_in, valid_out = split_annotated_data(\
                                                    input_statements, desired_statements)
    y_predicted_train = [statement for statement in filter_model.sentence_filter(train_in)]
    y_predicted_valid = [statement for statement in filter_model.sentence_filter(valid_in)]
    compute_filtering_metrics(train_out, y_predicted_train)
    compute_filtering_metrics(valid_out, y_predicted_valid)



def write_single_column_file_to_csv(input_file_name: str, output_file_name: str,
                                    number_of_columns=3, shuffle=False):
    """
    This is a function just for data format manipulation.
    Given an input_file_name, the function re-arranges the data and writes it to
    output_file_name. The input file is expected to have one paragraph/sentence
    per line. The output will be a csv file with multiple columns.
    Intention:
    The required Mechanical Turk format is a csv file with three columns,
    namely snippet1,snippet2,snippet3.
    """

    # read in data
    with open(input_file_name, 'r') as inputfile:
        items_list = [line.rstrip() for line in inputfile.readlines() if line]

    # trim until list length is divisible by the number of columns intended in csv file
    while len(items_list) % number_of_columns != 0:
        _ = items_list.pop()

    print("Writing to csv file: ", output_file_name,\
          "- Number of cells (csv body):", len(items_list),\
          "- number_of_columns:", number_of_columns)

    if shuffle:
        random.shuffle(items_list)

    # select slices (prospective csv columns) of the input list
    slices = []
    for column_index in range(number_of_columns):
        this_slice = items_list[column_index::number_of_columns]
        slices.append(this_slice)

    # write to csv file
    with open(output_file_name, 'w') as outputfile:
        writer = csv.writer(outputfile)
        # csv header
        writer.writerow(['snippet'+str(x) for x in range(1, number_of_columns+1, 1)])
        # csv body
        for row_index in range(len(slices[0])):
            row = [s[row_index] for s in slices]
            writer.writerow(row)



def main():
    """
    main function. Illustrates usage of the individual components.
    """

    #### define a sentence filter model

    # files containing male/female first names. These have to be specified.
    names_files = ['/Users/johannesw/Desktop/Johannes/dataset_creation/data/names/male.txt',
                   '/Users/johannesw/Desktop/Johannes/dataset_creation/data/names/female.txt']
    #names_files = ['/path/to/male.txt',
    #               '/path/to/female.txt']
    assert len(names_files) == 2

    # file containing list of discourse markers
    discourse_markers_file_name = 'discourse_markers.txt'

    filter_model = SentenceFilter(names_files, discourse_markers_file_name)


    #### load some text paragraphs
    # need to specify path, e.g.
    path = "/path/to/science_textbooks/"
    path = "/Users/johannesw/Desktop/Johannes/dataset_creation/data/science_textbooks/"
    list_of_textbook_filenames = [path + file_name for file_name in os.listdir(path)]

    textbook_paragraphs = load_paragraphs_from_files(list_of_textbook_filenames)


    #### can include a set of paragraphs that shouldn't be used (not required)
    # e.g. because these paragraphs were used already
    #turk_file = 'path/to/MTurk-output/download.csv'
    #used_up_paragraphs, bad_paragraphs = \
    #    identify_used_paragraphs(turk_file)
    #not_use_these_paragraphs = set(used_up_paragraphs).union(bad_paragraphs)
    not_use_these_paragraphs = set()

    #### apply the sentence filter model to the paragraphs
    chosen_paragraphs_file = 'chosen_paragraphs.txt'
    filter_paragraphs(textbook_paragraphs, not_use_these_paragraphs, \
                      filter_model, chosen_paragraphs_file)


    #### writing the outputs into 3-column csv format
    write_single_column_file_to_csv(input_file_name=chosen_paragraphs_file,
                                    output_file_name=chosen_paragraphs_file+'.csv',
                                    number_of_columns=3, shuffle=True)


if __name__ == '__main__':
    main()
