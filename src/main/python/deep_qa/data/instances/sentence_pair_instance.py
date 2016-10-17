from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer
from ..tokenizer import tokenizers, Tokenizer


class SentencePairInstance(TextInstance):
    """
    SentencePairInstance contains a labeled pair of instances accompanied by a binary label.  You
    could have the label represent whatever you want, such as entailment, or occuring in the same
    context, or whatever.
    """
    def __init__(self,
                 first_sentence: str,
                 second_sentence: str,
                 label: List[int],
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        super(SentencePairInstance, self).__init__(label, index, tokenizer)
        self.first_sentence = first_sentence
        self.second_sentence = second_sentence

    @overrides
    def words(self) -> List[str]:
        return self._tokenize(self.first_sentence.lower()) + self._tokenize(self.second_sentence.lower())

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        first_sentence = [data_indexer.get_word_index(word) for word in
                          self._tokenize(self.first_sentence.lower())]
        second_sentence = [data_indexer.get_word_index(word) for word in
                           self._tokenize(self.second_sentence.lower())]
        return IndexedSentencePairInstance(first_sentence, second_sentence, self.label, self.index)

    @classmethod
    def read_from_line(cls,
                       line: str,
                       default_label: bool=None,
                       tokenizer: Tokenizer=tokenizers['default']()):
        """
        Expected format:
        [sentence1][tab][sentence2][tab][label]
        """
        fields = line.split("\t")
        first_sentence, second_sentence, label = fields
        return cls(first_sentence, second_sentence, [int(label)], tokenizer)


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
    def get_lengths(self) -> Dict[str, int]:
        max_length = max(len(self.first_sentence_indices), len(self.second_sentence_indices))
        return {'word_sequence_length': max_length}

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) self.word_indices to be of length max_lengths[0].  See comment on
        self.get_lengths() for why max_lengths is a list instead of an int.
        """
        sentence_length = max_lengths['word_sequence_length']
        self.first_sentence_indices = self.pad_word_sequence_to_length(self.first_sentence_indices,
                                                                       sentence_length)
        self.second_sentence_indices = self.pad_word_sequence_to_length(self.second_sentence_indices,
                                                                        sentence_length)

    @overrides
    def as_training_data(self):
        first_sentence_array = numpy.asarray(self.first_sentence_indices, dtype='int32')
        second_sentence_array = numpy.asarray(self.second_sentence_indices, dtype='int32')
        return (first_sentence_array, second_sentence_array), numpy.asarray(self.label)
