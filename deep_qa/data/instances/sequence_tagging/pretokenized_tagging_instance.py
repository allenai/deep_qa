from typing import List

import numpy
from overrides import overrides

from .tagging_instance import TaggingInstance
from ...data_indexer import DataIndexer

class PreTokenizedTaggingInstance(TaggingInstance):
    """
    This is a ``TaggingInstance`` where the text has been pre-tokenized.  Thus the ``text`` member
    variable here is actually a ``List[str]``, instead of a ``str``.

    When using this ``Instance``, you `must` use the ``NoOpWordSplitter`` as well, or things will
    break.  You probably also do not want any kind of filtering (though stemming is ok), because
    only the words will get filtered, not the labels.
    """
    def __init__(self, text: List[str], label: List[str], index: int=None):
        super(PreTokenizedTaggingInstance, self).__init__(text, label, index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str, default_label: bool=None):
        """
        Reads a ``PreTokenizedTaggingInstance`` from a line.  The format has one of two options:

        1. [example index][token1]###[tag1][tab][token2]###[tag2][tab]...
        2. [token1]###[tag1][tab][token2]###[tag2][tab]...

        default_label is ignored, but we keep the argument to match the interface.
        """
        fields = line.split("\t")

        if fields[0].isdigit():
            index = int(fields[0])
            fields = fields[1:]
        else:
            index = None
        tokens = []
        tags = []
        for field in fields:
            token, tag = field.split("###")
            tokens.append(token)
            tags.append(tag)
        return cls(tokens, tags, index)

    @overrides
    def tags_in_label(self):
        return [tag for tag in self.label]

    @overrides
    def _index_label(self, label: List[str], data_indexer: DataIndexer) -> List[int]:
        tag_indices = [data_indexer.get_word_index(tag, namespace='tags') for tag in label]
        indexed_label = []
        for tag_index in tag_indices:
            # We subtract 2 here to account for the unknown and padding tokens that the DataIndexer
            # uses.
            tag_one_hot = numpy.zeros(data_indexer.get_vocab_size(namespace='tags') - 2)
            tag_one_hot[tag_index - 2] = 1
            indexed_label.append(tag_one_hot)
        return indexed_label
