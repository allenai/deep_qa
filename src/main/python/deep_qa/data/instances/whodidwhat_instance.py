from typing import Dict, List, Tuple
import numpy as np

from overrides import overrides
from .question_passage_instance import IndexedQuestionPassageInstance, QuestionPassageInstance
from ..data_indexer import DataIndexer
from ..tokenizer import tokenizers, Tokenizer


class WhoDidWhatInstance(QuestionPassageInstance):
    """
    A WhoDidWhatInstance is a QuestionPassageInstance that represents a (question,
    passage, answer_options) tuple from the WhoDidWhatInstance dataset, with an
    associated label indicating the index of the correct answer choice.
    """
    def __init__(self,
                 question: str,
                 passage: str,
                 answer_options: List[str],
                 label: int,
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        super(WhoDidWhatInstance, self).__init__(question, passage, label, index, tokenizer)
        self.answer_options = answer_options

    def __str__(self):
        return ('WhoDidWhatInstance({}, {}, {}, {})'.format(self.question_text,
                                                            self.passage_text,
                                                            '|'.join(self.answer_options),
                                                            str(self.label)))

    @overrides
    def words(self) -> List[str]:
        words = []
        words.extend(self._words_from_text(self.question_text))
        words.extend(self._words_from_text(self.passage_text))
        for option in self.answer_options:
            words.extend(self._words_from_text(option))
        return words

    @overrides
    def _index_label(self, label: Tuple[int, int]) -> List[int]:
        """
        Specify how to index `self.label`, which is needed to convert the
        WhoDidWhatInstance into an IndexedInstance (conversion handled in superclass).
        """
        return self.label

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        question_indices = self._index_text(self.question_text, data_indexer)
        passage_indices = self._index_text(self.passage_text, data_indexer)
        option_indices = [self._index_text(option, data_indexer) for option in
                          self.answer_options]
        return IndexedWhoDidWhatInstance(question_indices, passage_indices,
                                         option_indices, self.label, self.index)

    @classmethod
    def read_from_line(cls,
                       line: str,
                       default_label: bool=None,
                       tokenizer: Tokenizer=tokenizers['default']()):
        """
        Reads a WhoDidWhatInstance object from a line.  The format has one of two options:

        (1) [example index][tab][passage][tab][left_context][tab][right_context][tab][options][tab][label]
        (2) [passage][tab][left_context][tab][right_context][tab][options][tab][label]

        To form the cloze-style question, [left_context] is concatenated with "XXX",
        and then with [right_context]. "XXX" is the blank to be filled with one
        of the answer options.

        The `answer_options` column is assumed formatted as: [option]###[option]###[option]...
        That is, we split on three hashes ("###").

        default_label is ignored, but we keep the argument to match the interface.
        """
        fields = line.split("\t")

        if len(fields) == 6:
            index_string, passage, left_context, right_context, options, label_string = fields
            index = int(index_string)
        elif len(fields) == 5:
            passage, left_context, right_context, options, label_string = fields
            index = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        # get the answer options
        answer_options = options.split("###")
        label = int(label_string)

        # form the question from left and right contexts, keeping spacing consistent
        question = "{} XXX {}".format(left_context, right_context).strip()

        return cls(question, passage, answer_options, label, index, tokenizer)


class IndexedWhoDidWhatInstance(IndexedQuestionPassageInstance):
    def __init__(self,
                 question_indices: List[int],
                 passage_indices: List[int],
                 option_indices: List[List[int]],
                 label: List[int],
                 index: int=None):
        super(IndexedWhoDidWhatInstance, self).__init__(question_indices, passage_indices,
                                                        label, index)
        self.option_indices = option_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedQuestionPassageInstance([], [], [[]], None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        We need to pad the answer option length (in words), the number of answer
        options, the question length (in words), the passage length (in words),
        and the word length (in characters) among all the questions, passages,
        and answer options.
        """
        option_lengths = [self._get_word_sequence_lengths(option) for option in self.option_indices]

        lengths = super(IndexedWhoDidWhatInstance, self).get_lengths()

        # the number of options
        lengths['num_options'] = len(self.option_indices)

        # the number of words in the longest option
        lengths['num_option_words'] = max([lengths['word_sequence_length'] for
                                           lengths in option_lengths])
        # the length of the longest word across the passage, question, and options
        if 'word_character_length' in option_lengths[0]:
            # length of longest word (in characters) in options
            max_option_word_length = max([lengths['word_character_length'] for
                                          lengths in option_lengths])

            lengths['word_character_length'] = max(lengths['word_character_length'],
                                                   max_option_word_length)

        return lengths

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        In this function, we pad the questions and passages (in terms of number of words in each),
        as well as the individual words in the questions and passages themselves. We also pad the
        number of answer options, the answer options (in terms of numbers or words in each),
        as well as the individual words in the answer options.
        """
        super(IndexedWhoDidWhatInstance, self).pad(max_lengths)

        # pad the number of options
        num_options = max_lengths['num_options']
        while len(self.option_indices) < num_options:
            self.option_indices.append([])
        self.option_indices = self.option_indices[:num_options]

        # pad the number of words in the options, number of characters in each word in option
        padded_options = []
        for indices in self.option_indices:
            max_lengths['word_sequence_length'] = max_lengths['num_option_words']
            padded_options.append(self.pad_word_sequence(indices, max_lengths))
        self.option_indices = padded_options


    @overrides
    def as_training_data(self):
        question_array = np.asarray(self.question_indices, dtype='int32')
        passage_array = np.asarray(self.passage_indices, dtype='int32')
        options_array = np.asarray(self.option_indices, dtype='int32')
        if self.label is None:
            label = None
        else:
            label = np.zeros((len(self.option_indices)))
            label[self.label] = 1
        return (question_array, passage_array, options_array), label
