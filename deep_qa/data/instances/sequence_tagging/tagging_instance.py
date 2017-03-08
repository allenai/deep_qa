from typing import Dict, List, Any

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class TaggingInstance(TextInstance):
    """
    A ``TaggingInstance`` represents a passage of text and a tag sequence over that text.

    There are some sticky issues with tokenization and how exactly the label is specified.  For
    example, if your label is a sequence of tags, that assumes a particular tokenization, which
    interacts in a funny way with our tokenization code.  This is a general superclass containing
    common functionality for most simple sequence tagging tasks.  The specifics of reading in data
    from a file and converting that data into properly-indexed tag sequences is left to subclasses.
    """
    def __init__(self, text: str, label: Any, index: int=None):
        super(TaggingInstance, self).__init__(label, index)
        self.text = text

    def __str__(self):
        return "TaggedSequenceInstance(" + self.text + ", " + str(self.label) + ")"

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.text)
        words['tags'] = self.tags_in_label()
        return words

    def tags_in_label(self):
        """
        Returns all of the tag words in this instance, so that we can convert them into indices.
        This is called in ``self.words()``.  Not necessary if you have some pre-indexed labeling
        scheme.
        """
        raise NotImplementedError

    def _index_label(self, label: Any, data_indexer: DataIndexer) -> List[int]:
        """
        Index the labels. Since we don't know what form the label takes, we leave it to subclasses
        to implement this method.  If you need to convert tag names into indices, use the namespace
        'tags' in the ``DataIndexer``.
        """
        raise NotImplementedError

    def to_indexed_instance(self, data_indexer: DataIndexer):
        text_indices = self._index_text(self.text, data_indexer)
        label_indices = self._index_label(self.label, data_indexer)
        assert len(text_indices) == len(label_indices), "Tokenization is off somehow"
        return IndexedTaggingInstance(text_indices, label_indices, self.index)


class IndexedTaggingInstance(IndexedInstance):
    def __init__(self, text_indices: List[int], label: List[int], index: int=None):
        super(IndexedTaggingInstance, self).__init__(label, index)
        self.text_indices = text_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return TaggingInstance([], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        return self._get_word_sequence_lengths(self.text_indices)

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        self.text_indices = self.pad_word_sequence(self.text_indices, max_lengths,
                                                   truncate_from_right=False)

    @overrides
    def as_training_data(self):
        text_array = numpy.asarray(self.text_indices, dtype='int32')
        label_array = numpy.asarray(self.label, dtype='int32')
        return text_array, label_array
