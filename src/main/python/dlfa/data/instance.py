from typing import List

import numpy
from nltk.tokenize import word_tokenize

from .index_data import DataIndexer

class Instance(object):
    """
    A data instance, used either for training a neural network or for testing one.
    """
    def __init__(self, text: str, label: bool, index: int=None):
        """
        text: the text of this instance, either a sentence or a logical form.
        label: True, False, or None, indicating whether the instance is a positive, negative or
            unknown (i.e., test) example, respectively.
        index: if given, must be an integer.  Used for matching instances with other data, such as
            background sentences.
        """
        self.text = text
        self.label = label
        self.index = index

    def to_indexed_instance(self, data_indexer: DataIndexer):
        words = word_tokenize(self.text.lower())
        indices = [data_indexer.get_word_index(word) for word in words]
        return IndexedInstance(indices, self.label, self.index)

    @staticmethod
    def read_from_line(line: str, default_label: bool=None):
        """
        Reads an Instance object from a line.  The format has one of four options:

        (1) [sentence]
        (2) [sentence index][tab][sentence]
        (3) [sentence][tab][label]
        (4) [sentence index][tab][sentence][tab][label]

        For options (1) and (2), we use the default_label to give the Instance a label, and for
        options (3) and (4), we check that default_label matches the label in the file, if
        default_label is given.
        """
        fields = line.split("\t")
        if len(fields) == 3:
            index, text, label_string = fields
            label = label_string == '1'
            Instance._check_label(label, default_label)
            return Instance(text, label, int(index))
        elif len(fields) == 2:
            if fields[0].isdecimal():
                index, text = fields
                Instance._check_label(None, default_label)
                return Instance(text, default_label, int(index))
            elif fields[1].isdecimal():
                text, label_string = fields
                label = label_string == '1'
                Instance._check_label(label, default_label)
                return Instance(text, label)
            else:
                raise RuntimeError("Unrecognized line format: " + line)
        elif len(fields) == 1:
            text = fields[0]
            Instance._check_label(None, default_label)
            return Instance(text, default_label)
        else:
            raise RuntimeError("Unrecognized line format: " + line)

    @staticmethod
    def _check_label(label: bool, default_label: bool):
        if default_label is not None and label is not None and label != default_label:
            raise RuntimeError("Default label given with file, and label in file doesn't match!")

    def as_training_data(self):  # pylint: disable=no-self-use
        raise RuntimeError("Must index an instance before it can be made training data!")


class IndexedInstance(object):
    """
    An indexed data instance, which is a list of word indices, coupled with a label, and possibly
    an instance index.  An IndexedInstance is created from an Instance using a DataIndexer, and the
    indices here have no recoverable meaning without the DataIndexer.

    For example, we might have the following instance:
        Instance('Jamie is nice, Holly is mean', True, 25).
    After being converted into an IndexedInstance, we might have the following:
        IndexedInstance([1, 6, 7, 1, 6, 8], True, 25).
    This would mean that "Jamie" and "Holly" were OOV to the DataIndexer, and the other words were
    given indices.
    """
    def __init__(self, word_indices: List[int], label: bool, index: int=None):
        self.word_indices = word_indices
        self.label = label
        self.index = index

    def get_lengths(self) -> List[int]:
        """
        This is prep for padding.  We want to know, for all parts of this instance that are going
        to be converted to arrays, what the length of this instance is.  That way we can find the
        max, then pad all of the instances to the same length.  This simple IndexedInstance only
        has one such length (for word_indices), but more complex Instance objects might have
        several lengths, so we're going to return a list here.
        """
        return [len(self.word_indices)]

    def pad(self, max_lengths: List[int]):
        """
        Pads (or truncates) self.word_indices to be of length max_lengths[0].  See comment on
        self.get_lengths() for why this is a list instead of an int.
        """
        desired_length = max_lengths[0]
        padded_word_indices = [0] * desired_length
        indices_length = min(len(self.word_indices), desired_length)
        if indices_length != 0:
            padded_word_indices[-indices_length:] = self.word_indices[-indices_length:]
        self.word_indices = padded_word_indices

    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        label = numpy.zeros((2))
        if self.label is True:
            label[1] = 1
        elif self.label is False:
            label[0] = 1
        else:
            raise RuntimeError("Cannot make training data out of instances without labels!")
        return word_array, label


class BackgroundInstance(Instance):
    """
    An Instance that has background knowledge associated with it.
    """
    def __init__(self, text: str, background: List[str], label: bool, index: int=None):
        super(BackgroundInstance, self).__init__(text, label, index)
        self.background = background

    def to_indexed_instance(self, data_indexer: DataIndexer):
        words = word_tokenize(self.text.lower())
        word_indices = [data_indexer.get_word_index(word) for word in words]
        background_indices = []
        for text in self.background:
            words = word_tokenize(text.lower())
            indices = [data_indexer.get_word_index(word) for word in words]
            background_indices.append(indices)
        return IndexedBackgroundInstance(word_indices, background_indices, self.label, self.index)


class IndexedBackgroundInstance(IndexedInstance):
    """
    An IndexedInstance that has background knowledge associated with it, where the background
    knowledge has also been indexed.
    """
    def __init__(self,
                 word_indices: List[int],
                 background_indices: List[List[int]],
                 label: bool,
                 index: int=None):
        super(IndexedBackgroundInstance, self).__init__(word_indices, label, index)
        self.background_indices = background_indices

    def get_lengths(self) -> List[int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for background_indices.

        Additionally, as we currently use the same encoder for both a sentence and its background
        knowledge, we'll also modify the word_indices length to look at the background sentences
        too.
        """
        lengths = super(IndexedBackgroundInstance, self).get_lengths()
        lengths.extend([len(self.background_indices)])
        if self.background_indices:
            max_background_length = max(len(background) for background in self.background_indices)
            lengths[0] = max(lengths[0], max_background_length)
        return lengths

    def pad(self, max_lengths: List[int]):
        """
        We let the super class deal with padding word_indices; we'll worry about padding
        background_indices.  We need to pad it in two ways: (1) we need len(background_indices) to
        be the same for all instances, and (2) we need len(background_indices[i]) to be the same
        for all i, for all instances.  We'll use the word_indices length from the super class for
        (2).  """
        super(IndexedBackgroundInstance, self).pad(max_lengths)
        word_sequence_length = max_lengths[0]

        # We need to do this to find out what index we added our length at.  We could just do
        # lengths[1] to get the background length, but this future proofs the code, in case things
        # change in the super class.
        super_lengths = super(IndexedBackgroundInstance, self).get_lengths()
        background_length_index = len(super_lengths)
        background_length = max_lengths[background_length_index]

        # Padding (1): making sure we have the right number of background sentences.
        for _ in range(background_length - len(self.background_indices)):
            self.background_indices.append([[0]])

        # Padding (2): making sure all background sentences have the right length.
        padded_background = []
        for background in self.background_indices:
            padded_word_indices = [0] * word_sequence_length
            indices_length = min(len(background), word_sequence_length)
            if indices_length != 0:
                padded_word_indices[-indices_length:] = background[-indices_length:]
            padded_background.append(padded_word_indices)
        self.background_indices = padded_background

    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        background_array = numpy.asarray(self.background_indices, dtype='int32')
        label = numpy.zeros((2))
        if self.label is True:
            label[1] = 1
        elif self.label is False:
            label[0] = 1
        else:
            raise RuntimeError("Cannot make training data out of instances without labels!")
        return (word_array, background_array), label
