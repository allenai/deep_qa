from collections import defaultdict
from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer


class TupleInstance(TextInstance):
    """
    A TupleInstance is a kind of TextInstance that has text in multiple slots. This can be used to
    store SVO triples.
    """
    def __init__(self, text: List[str], label: bool=None, index: int=None):
        """
        text: list of phrases that form the tuple. The first two slots are subject and verb, and the
            remaining are objects.
        """
        super(TupleInstance, self).__init__(label, index)
        self.text = text

    def __str__(self):
        return 'TupleInstance( [' + ',\n'.join(self.text) + '] , ' + str(self.label) + ')'

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = defaultdict(list)
        for phrase in self.text:
            phrase_words = self._words_from_text(phrase)
            for namespace in phrase_words:
                words[namespace].extend(phrase_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = [self._index_text(phrase, data_indexer) for phrase in self.text]
        return IndexedTupleInstance(indices, self.label, self.index)

    @classmethod
    def read_from_line(cls, line: str, default_label: bool=True):
        """
        Reads a TupleInstances from a line.  The format has one of four options:

        (1) [subject]###[predicate]###[object1]...
        (2) [sentence index][tab][subject]###[predicate]###[object1]...
        (3) [subject]###[predicate]###[object1]...[tab][label]
        (4) [sentence index][tab][subject]###[predicate]###[object1]...[tab][label]

        Objects are optional, and can vary in number. This makes the acceptable number of slots per
        tuple, 2 or more.
        """
        fields = line.split("\t")
        if len(fields) == 1:
            # Case 1
            tuple_string = fields[0]
            index = None
            label = default_label
        elif len(fields) == 2:
            if fields[0].isdigit():
                # Case 2
                index = int(fields[0])
                tuple_string = fields[1]
                label = default_label
            else:
                # Case 3
                tuple_string = fields[0]
                index = None
                label = fields[2].strip() == "1"
        else:
            # Case 4
            index = int(fields[0])
            tuple_string = fields[1]
            label = fields[2].strip() == "1"
        tuple_fields = tuple_string.split('###')
        if len(tuple_fields) < 2:
            raise RuntimeError("Unexpected number of fields in tuple: " + tuple_string)
        return cls(tuple_fields, label=label, index=index)


class IndexedTupleInstance(IndexedInstance):
    def __init__(self, word_indices: List[List[int]], label, index: int=None):
        super(IndexedTupleInstance, self).__init__(label, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedTupleInstance([], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        # We care only about the longest slot here because all slots will be padded to the same length.
        return {'num_sentence_words': max([len(indices) for indices in self.word_indices]),
                'num_slots': len(self.word_indices)}

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) all slots in self.word_indices to the same length. Additionally,
        adjusts the number of slots in the tuple to the desired number: if there are fewer
        slots than desired, will add empty lists at the end, and if there are more will
        concatenate object slots.
        Note: Adjusting the number of slots happens before adjusting the lengths of all the slots.
        since truncating is done from the left, the earlier object slots ar more likely to be
        lost.
        """
        desired_num_slots = max_lengths['num_slots']
        if len(self.word_indices) > desired_num_slots:
            # We concatenate the indices from the slots at the end to make word_indices the right
            # length.
            num_extra_slots = len(self.word_indices) - desired_num_slots
            all_word_indices = self.word_indices
            self.word_indices = all_word_indices[:desired_num_slots]
            for i in range(num_extra_slots):
                self.word_indices[desired_num_slots - 1] += all_word_indices[desired_num_slots + i]
        elif len(self.word_indices) < desired_num_slots:
            additional_slots = [[] for _ in range(desired_num_slots - len(self.word_indices))]
            self.word_indices += additional_slots
        self.word_indices = [self.pad_word_sequence(indices, max_lengths) for
                             indices in self.word_indices]

    @overrides
    def as_training_data(self):
        # (num_slots, max_length)
        tuple_matrix = numpy.asarray(self.word_indices, dtype='int32')
        if self.label is True:
            label = numpy.zeros((2))
            label[1] = 1
        elif self.label is False:
            label = numpy.zeros((2))
            label[0] = 1
        else:
            label = None
        return tuple_matrix, label
