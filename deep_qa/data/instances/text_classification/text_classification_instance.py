from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class TextClassificationInstance(TextInstance):
    """
    A TextClassificationInstance is a :class:`TextInstance` that is a single passage of text,
    where that passage has some associated (categorical, or possibly real-valued) label.
    """
    def __init__(self, text: str, label: bool, index: int=None):
        """
        text: the text of this instance, typically either a sentence or a logical form.
        """
        super(TextClassificationInstance, self).__init__(label, index)
        self.text = text

    def __str__(self):
        return 'TextClassificationInstance(' + self.text + ', ' + str(self.label) + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        return self._words_from_text(self.text)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = self._index_text(self.text, data_indexer)
        return IndexedTextClassificationInstance(indices, self.label, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads a TextClassificationInstance object from a line.  The format has one of four options:

        (1) [sentence]
        (2) [sentence index][tab][sentence]
        (3) [sentence][tab][label]
        (4) [sentence index][tab][sentence][tab][label]

        If no label is given, we use ``None`` as the label.
        """
        fields = line.split("\t")

        if len(fields) == 3:
            index, text, label_string = fields
            label = label_string == '1'
            return cls(text, label, int(index))
        elif len(fields) == 2:
            if fields[0].isdecimal():
                index, text = fields
                return cls(text, None, int(index))
            elif fields[1].isdecimal():
                text, label_string = fields
                label = label_string == '1'
                return cls(text, label)
            else:
                raise RuntimeError("Unrecognized line format: " + line)
        elif len(fields) == 1:
            text = fields[0]
            return cls(text, None)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class IndexedTextClassificationInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], label, index: int=None):
        super(IndexedTextClassificationInstance, self).__init__(label, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedTextClassificationInstance([], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        This simple IndexedInstance only has one padding dimension: word_indices.
        """
        return self._get_word_sequence_lengths(self.word_indices)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        self.word_indices = self.pad_word_sequence(self.word_indices, padding_lengths)

    @overrides
    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        if self.label is True:
            label = numpy.zeros((2))
            label[1] = 1
        elif self.label is False:
            label = numpy.zeros((2))
            label[0] = 1
        else:
            label = None
        return word_array, label
