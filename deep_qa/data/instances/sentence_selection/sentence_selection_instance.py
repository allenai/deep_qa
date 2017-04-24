from typing import Dict, List

import numpy as np
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class SentenceSelectionInstance(TextInstance):
    """
    A SentenceSelectionInstance is an instance for the sentence selection
    task. A SentenceSelectionInstance stores a question as a string, and a set of sentences
    as a list of strings. The labels is a single int, indicating the index of
    the sentence that contains the answer to the question.
    """
    def __init__(self, question_text: str, sentences: List[str], label: int, index: int=None):
        super(SentenceSelectionInstance, self).__init__(label, index)
        self.question_text = question_text
        self.sentences = sentences

    def __str__(self):
        return ('SentenceSelectionInstance(' + self.question_text +
                ', ' + self.sentences + ', ' +
                str(self.label) + ')')

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.question_text)
        sentences_words = [self._words_from_text(sentence) for sentence in
                           self.sentences]
        for namespace in words:
            for sentence_words in sentences_words:
                words[namespace].extend(sentence_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        question_indices = self._index_text(self.question_text, data_indexer)
        sentences_indices = [self._index_text(sentence, data_indexer)
                             for sentence in self.sentences]
        return IndexedSentenceSelectionInstance(question_indices,
                                                sentences_indices,
                                                self.label,
                                                self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads a SentenceSelectionInstance object from a line.
        The format has one of two options:

        (1) [example index][tab][question][tab][list_of_sentences][tab][label]
        (2) [question][tab][list_of_sentences][tab][label]

        The ``list_of_sentences`` column is assumed formatted as: ``[sentence]###[sentence]###[sentence]...``
        That is, we split on three hashes (``"###"``).
        """
        fields = line.split("\t")

        if len(fields) == 4:
            index_string, question, sentences, label_string = fields
            index = int(index_string)
        elif len(fields) == 3:
            question, sentences, label_string = fields
            index = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        sentences_split = sentences.split("###")
        label = int(label_string)

        return cls(question, sentences_split, label, index)


class IndexedSentenceSelectionInstance(IndexedInstance):
    """
    This is an indexed instance that is used for the answer sentence selection task,
    where we have a question, a list of candidate sentences, and an
    int label that indicates the index of the correct sentence in the list of
    candidate sentences.
    """
    def __init__(self,
                 question_indices: List[int],
                 sentences_indices: List[List[int]],
                 label: int,
                 index: int=None):
        super(IndexedSentenceSelectionInstance, self).__init__(label, index)
        self.question_indices = question_indices
        self.sentences_indices = sentences_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedSentenceSelectionInstance([], [], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        We need to pad at least the question length, the passage length, and the
        word length across all the questions and passages. Subclasses that
        add more arguments should also override this method to enable padding on said
        arguments.
        """
        question_lengths = self._get_word_sequence_lengths(self.question_indices)
        sentences_lengths = [self._get_word_sequence_lengths(sentence_indices)
                             for sentence_indices in self.sentences_indices]
        lengths = {}

        # the number of sentences
        lengths['num_sentences'] = len(self.sentences_indices)

        # the number of words in the question
        lengths['num_question_words'] = question_lengths['num_sentence_words']

        # the number of words in each passage
        lengths['num_sentence_words'] = max(sentence_lengths['num_sentence_words']
                                            for sentence_lengths in sentences_lengths)

        if 'num_word_characters' in question_lengths and 'num_word_characters' in sentences_lengths[0]:
            # the length of the longest word across all the sentences and the question
            lengths['num_word_characters'] = max(question_lengths['num_word_characters'],
                                                 max(sentence_lengths['num_word_characters']
                                                     for sentence_lengths in sentences_lengths))
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        In this function, we pad the questions and sentences (in terms of number
        of words in each), the number of sentences,
        as well as the individual words in the questions and sentences themselves.
        """
        padding_lengths_tmp = padding_lengths.copy()
        # Pad the number of sentences.
        num_sentences = padding_lengths['num_sentences']
        self.sentences_indices = self.pad_sequence_to_length(self.sentences_indices,
                                                             num_sentences,
                                                             lambda: [],
                                                             truncate_from_right=False)

        # Pad the number of words in a sentence.
        # Since the number of words in the sentence is set
        # to ['num_sentence_words'] (the key defined in the TextTrainer)
        # by default, we don't have to modify it.
        self.sentences_indices = [self.pad_word_sequence(sentence_indices,
                                                         padding_lengths_tmp)
                                  for sentence_indices in self.sentences_indices]
        # Pad the number of words in a question.
        padding_lengths_tmp['num_sentence_words'] = padding_lengths_tmp['num_question_words']
        self.question_indices = self.pad_word_sequence(self.question_indices, padding_lengths_tmp)

    @overrides
    def as_training_data(self):
        question_array = np.asarray(self.question_indices, dtype='int32')
        sentences_matrix = np.asarray(self.sentences_indices, dtype='int32')
        if self.label is None:
            label = None
        else:
            label = np.zeros((len(self.sentences_indices)))
            label[self.label] = 1
        return (question_array, sentences_matrix), np.asarray(label)
