from typing import List

import numpy

class IndexedInstance:
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
        This simple IndexedInstance only has one padding dimension: word_indices.
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


class IndexedLogicalFormInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], transitions: List[int], label: bool, index: int=None):
        super(IndexedLogicalFormInstance, self).__init__(word_indices, label, index)
        self.transitions = transitions

    def get_lengths(self) -> List[int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for `transitions`.
        """
        lengths = super(IndexedLogicalFormInstance, self).get_lengths()
        lengths.append(len(self.transitions))
        return lengths

    def pad(self, max_lengths: List[int]):
        """
        We let the super class deal with padding word_indices; we'll worry about padding
        transitions.
        """
        super(IndexedLogicalFormInstance, self).pad(max_lengths)

        # We need to do this to find out what index we added our length at.  We could just do
        # lengths[1] to get the transitions length, but this future-proofs the code, in case things
        # change in the super class.  lengths[-1] doesn't work either, because this could be
        # sub-classed, and then the subclass may have added another value to the lengths.
        super_lengths = super(IndexedLogicalFormInstance, self).get_lengths()
        transition_length_index = len(super_lengths)
        transition_length = max_lengths[transition_length_index]

        padded_transitions = [0] * transition_length
        indices_length = min(len(self.transitions), transition_length)
        if indices_length != 0:
            padded_transitions[-indices_length:] = self.transitions[-indices_length:]
        self.transitions = padded_transitions

    def as_training_data(self):
        word_array, label = super(IndexedLogicalFormInstance, self).as_training_data()
        transitions = numpy.asarray(self.transitions, dtype='int32')
        return (word_array, transitions), label


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
        lengths.append(len(self.background_indices))
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
        (2).
        """
        super(IndexedBackgroundInstance, self).pad(max_lengths)
        word_sequence_length = max_lengths[0]

        # We need to do this to find out what index we added our length at.  We could just do
        # lengths[1] to get the background length, but this future proofs the code, in case things
        # change in the super class.
        super_lengths = super(IndexedBackgroundInstance, self).get_lengths()
        background_length_index = len(super_lengths)
        background_length = max_lengths[background_length_index]

        # Padding (1): making sure we have the right number of background sentences.  We also need
        # to truncate, if necessary.
        if len(self.background_indices) > background_length:
            self.background_indices = self.background_indices[:background_length]
        for _ in range(background_length - len(self.background_indices)):
            self.background_indices.append([0])

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
        word_array, label = super(IndexedBackgroundInstance, self).as_training_data()
        background_array = numpy.asarray(self.background_indices, dtype='int32')
        return (word_array, background_array), label
