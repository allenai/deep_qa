from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class SentencePairInstance(TextInstance):
    """
    SentencePairInstance contains a labeled pair of instances accompanied by a binary label.  You
    could have the label represent whatever you want, such as entailment, or occuring in the same
    context, or whatever.
    """
    def __init__(self, first_sentence: str, second_sentence: str, label: List[int], index: int=None):
        super(SentencePairInstance, self).__init__(label, index)
        self.first_sentence = first_sentence
        self.second_sentence = second_sentence

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.first_sentence)
        second_sentence_words = self._words_from_text(self.second_sentence)
        for namespace in words:
            words[namespace].extend(second_sentence_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        first_sentence = self._index_text(self.first_sentence, data_indexer)
        second_sentence = self._index_text(self.second_sentence, data_indexer)
        return IndexedSentencePairInstance(first_sentence, second_sentence, self.label, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Expected format:
        [sentence1][tab][sentence2][tab][label]
        """
        fields = line.split("\t")
        first_sentence, second_sentence, label = fields
        return cls(first_sentence, second_sentence, [int(label)])


class IndexedSentencePairInstance(IndexedInstance):
    """
    This is an indexed instance that is commonly used for labeled sentence pairs. Examples of this are
    SnliInstances where we have a labeled pair of text and hypothesis, and a sentence2vec instance where the
    objective is to train an encoder to predict whether the sentences are in context or not.
    """
    def __init__(self, first_sentence_indices: List[int], second_sentence_indices: List[int], label: List[int],
                 index: int=None):
        super(IndexedSentencePairInstance, self).__init__(label, index)
        self.first_sentence_indices = first_sentence_indices
        self.second_sentence_indices = second_sentence_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedSentencePairInstance([], [], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        first_sentence_lengths = self._get_word_sequence_lengths(self.first_sentence_indices)
        second_sentence_lengths = self._get_word_sequence_lengths(self.second_sentence_indices)
        lengths = {}
        for key in first_sentence_lengths:
            lengths[key] = max(first_sentence_lengths[key], second_sentence_lengths[key])
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        self.first_sentence_indices = self.pad_word_sequence(self.first_sentence_indices, padding_lengths)
        self.second_sentence_indices = self.pad_word_sequence(self.second_sentence_indices, padding_lengths)

    @overrides
    def as_training_data(self):
        first_sentence_array = numpy.asarray(self.first_sentence_indices, dtype='int32')
        second_sentence_array = numpy.asarray(self.second_sentence_indices, dtype='int32')
        return (first_sentence_array, second_sentence_array), numpy.asarray(self.label)
