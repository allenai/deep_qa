from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer
from ..tokenizer import tokenizers, Tokenizer


class TrueFalseInstance(TextInstance):
    """
    A TrueFalseInstance is a TextInstance that is a statement, where the statement is either true
    or false.
    """
    def __init__(self,
                 text: str,
                 label: bool,
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        """
        text: the text of this instance, typically either a sentence or a logical form.
        """
        super(TrueFalseInstance, self).__init__(label, index, tokenizer)
        self.text = text

    def __str__(self):
        return 'TrueFalseInstance(' + self.text + ', ' + str(self.label) + ')'

    @overrides
    def words(self) -> List[str]:
        return self._words_from_text(self.text)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = self._index_text(self.text, data_indexer)
        return IndexedTrueFalseInstance(indices, self.label, self.index)

    @classmethod
    def read_from_line(cls,
                       line: str,
                       default_label: bool=None,
                       tokenizer: Tokenizer=tokenizers['default']()):
        """
        Reads a TrueFalseInstance object from a line.  The format has one of four options:

        (1) [sentence]
        (2) [sentence index][tab][sentence]
        (3) [sentence][tab][label]
        (4) [sentence index][tab][sentence][tab][label]

        For options (1) and (2), we use the default_label to give the Instance a label, and for
        options (3) and (4), we check that default_label matches the label in the file, if
        default_label is given.

        The reason we check for a match between the read label and the default label in cases (3)
        and (4) is that if you passed a default label, you should be confident that everything
        you're reading has that label.  If we find one that doesn't match, you probably messed up
        some parameters somewhere else in your code.
        """
        fields = line.split("\t")

        # We'll call Instance._check_label for all four cases, even though it means passing None to
        # two of them.  We do this mainly for consistency, and in case the _check_label() ever
        # changes to actually do something with the label=None case.
        if len(fields) == 3:
            index, text, label_string = fields
            label = label_string == '1'
            cls._check_label(label, default_label)
            return cls(text, label, int(index), tokenizer)
        elif len(fields) == 2:
            if fields[0].isdecimal():
                index, text = fields
                cls._check_label(None, default_label)
                return cls(text, default_label, int(index), tokenizer)
            elif fields[1].isdecimal():
                text, label_string = fields
                label = label_string == '1'
                cls._check_label(label, default_label)
                return cls(text, label, tokenizer=tokenizer)
            else:
                raise RuntimeError("Unrecognized line format: " + line)
        elif len(fields) == 1:
            text = fields[0]
            cls._check_label(None, default_label)
            return cls(text, default_label, tokenizer=tokenizer)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class IndexedTrueFalseInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], label, index: int=None):
        super(IndexedTrueFalseInstance, self).__init__(label, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedTrueFalseInstance([], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        This simple IndexedInstance only has one padding dimension: word_indices.
        """
        return self._get_word_sequence_lengths(self.word_indices)

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) self.word_indices to be of length max_lengths[0].  See comment on
        self.get_lengths() for why max_lengths is a list instead of an int.
        """
        self.word_indices = self.pad_word_sequence(self.word_indices, max_lengths)

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
