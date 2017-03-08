from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer


class QuestionAnswerInstance(TextInstance):
    """
    A QuestionAnswerInstance has question text and a list of options, where one of those options is
    the answer to the question.  The question and answers are separate data structures and used as
    separate inputs to a model.  This differs from a MultipleChoiceInstance in that there is no
    associated question text in the MultipleChoiceInstance, just a list of true/false statements,
    one of which is true.
    """
    def __init__(self, question_text: str, answer_options: List[str], label: int, index: int=None):
        super(QuestionAnswerInstance, self).__init__(label, index)
        self.question_text = question_text
        self.answer_options = answer_options

    def __str__(self):
        return 'QuestionAnswerInstance(' + self.question_text + ', ' + \
                '|'.join(self.answer_options) + ', ' + str(self.label) + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.question_text)
        for option in self.answer_options:
            option_words = self._words_from_text(option)
            for namespace in words:
                words[namespace].extend(option_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        question_indices = self._index_text(self.question_text, data_indexer)
        option_indices = []
        for option in self.answer_options:
            indices = self._index_text(option, data_indexer)
            option_indices.append(indices)
        return IndexedQuestionAnswerInstance(question_indices, option_indices, self.label, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str, default_label: bool=None):
        """
        Reads a QuestionAnswerInstance object from a line.  The format has two options:

        (1) [question][tab][answer_options][tab][correct_answer]
        (2) [instance index][tab][question][tab][answer_options][tab][correct_answer]

        The `answer_options` column is assumed formatted as: [option]###[option]###[option]...
        That is, we split on three hashes ("###").

        default_label is ignored, but we keep the argument to match the interface.
        """
        fields = line.split("\t")

        if len(fields) == 3:
            question, answers, label_string = fields
            index = None
        elif len(fields) == 4:
            if fields[0].isdecimal():
                index_string, question, answers, label_string = fields
                index = int(index_string)
            else:
                raise RuntimeError("Unrecognized line format: " + line)
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        answer_options = answers.split("###")
        label = int(label_string)
        return cls(question, answer_options, label, index)


class IndexedQuestionAnswerInstance(IndexedInstance):
    def __init__(self,
                 question_indices: List[int],
                 option_indices: List[List[int]],
                 label: int,
                 index: int=None):
        super(IndexedQuestionAnswerInstance, self).__init__(label, index)
        self.question_indices = question_indices
        self.option_indices = option_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedQuestionAnswerInstance([], [], 0)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        At least three things to pad here: the question length, the answer option length, and the
        number of answer options.  There could also be a fourth: the character length of the words
        in the question and the answers.
        """
        question_lengths = self._get_word_sequence_lengths(self.question_indices)
        answer_lengths = [self._get_word_sequence_lengths(option) for option in self.option_indices]
        max_answer_length = max([lengths['num_sentence_words'] for lengths in answer_lengths])
        num_options = len(self.option_indices)
        lengths = {}
        lengths.update(question_lengths)
        if 'num_word_characters' in question_lengths:
            max_answer_character_length = max([lengths['num_word_characters'] for lengths in answer_lengths])
            max_character_length = max([question_lengths['num_word_characters'], max_answer_character_length])
            lengths['num_word_characters'] = max_character_length
        lengths['answer_length'] = max_answer_length
        lengths['num_options'] = num_options
        return lengths

    @overrides
    def pad(self, max_lengths: List[int]):
        """
        Three things to pad here: the question length, the answer option length, and the number of
        answer options.
        """
        self.question_indices = self.pad_word_sequence(self.question_indices, max_lengths)

        num_options = max_lengths['num_options']
        while len(self.option_indices) < num_options:
            self.option_indices.append([])
        self.option_indices = self.option_indices[:num_options]

        padded_options = []
        for indices in self.option_indices:
            answer_lengths = {}
            answer_lengths.update(max_lengths)
            answer_lengths['num_sentence_words'] = max_lengths['answer_length']
            padded_options.append(self.pad_word_sequence(indices, answer_lengths))
        self.option_indices = padded_options

    @overrides
    def as_training_data(self):
        question_array = numpy.asarray(self.question_indices, dtype='int32')
        option_array = numpy.asarray(self.option_indices, dtype='int32')
        if self.label is None:
            label = None
        else:
            label = numpy.zeros((len(self.option_indices)))
            label[self.label] = 1
        return (question_array, option_array), label
