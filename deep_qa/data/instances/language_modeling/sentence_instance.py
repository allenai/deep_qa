from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class SentenceInstance(TextInstance):
    """
    A ``SentenceInstance`` is a :class:`TextInstance` that is a single passage of text, with no
    associated label.  The label is the passage itself offset by one, because we will use this in a
    language modeling context, to predict the next word in the passage given the previous words.
    """
    def __init__(self, text: str, index: int=None):
        super(SentenceInstance, self).__init__(None, index)
        self.text = text

    def __str__(self):
        return 'SentenceInstance(' + self.text + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.text)
        words['words'].extend(['<S>', '</S>'])
        return self._words_from_text(self.text)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = self._index_text(self.text, data_indexer)
        # We'll add start and end symbols to the indices here, then split this into an input
        # sequence and an output sequence, offset by one, where the input has the start token, and
        # the output has the end token.
        start_index = data_indexer.get_word_index('<S>')
        end_index = data_indexer.get_word_index('</S>')
        if isinstance(indices[0], list):
            indices = [[start_index]] + indices + [[end_index]]
        else:
            indices = [start_index] + indices + [end_index]
        word_indices = indices[:-1]
        label_indices = indices[1:]
        if isinstance(label_indices[0], list):
            label_indices = [x[0] for x in label_indices]
        return IndexedSentenceInstance(word_indices, label_indices, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads a SentenceInstance object from a line.  The format has one of two options:

        (1) [sentence]
        (2) [sentence index][tab][sentence]
        """
        fields = line.split("\t")

        if len(fields) == 2:
            index, text = fields
            return cls(text, int(index))
        elif len(fields) == 1:
            text = fields[0]
            return cls(text, None)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class IndexedSentenceInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], label_indices: List[int], index: int=None):
        super(IndexedSentenceInstance, self).__init__(label_indices, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedSentenceInstance([], [], index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # len(label_indices) == len(word_indices), so we only need to return this one length.
        return self._get_word_sequence_lengths(self.word_indices)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        self.word_indices = self.pad_word_sequence(self.word_indices, padding_lengths)
        self.label = self.pad_sequence_to_length(self.label, padding_lengths['num_sentence_words'])

    @overrides
    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        label_array = numpy.asarray(self.label, dtype='int32')
        # The expand dims here is because Keras' sparse categorical cross entropy expects tensors
        # of shape (batch_size, num_words, 1).
        return word_array, numpy.expand_dims(label_array, axis=2)
