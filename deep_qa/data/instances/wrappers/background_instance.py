import codecs
from collections import OrderedDict
from typing import Dict, List

import numpy
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer
from ...dataset import TextDataset


def read_background_from_file(dataset: TextDataset, filename: str, background_class) -> TextDataset:
    """
    Reads a file formatted as background information and matches the background to the
    sentences in the given dataset.  The given dataset must have instance indices, so we can
    match the background information in the file to the instances in the dataset.

    The format for the file is assumed to be the following:
    [sentence index][tab][background 1][tab][background 2][tab][...]
    where [sentence index] corresponds to the index of one of the instances in `dataset`.

    This code will also work if the data is formatted simply as [index][tab][sentence], one per
    line.
    """
    new_instances = OrderedDict()
    for instance in dataset.instances:
        background_instance = BackgroundInstance(instance, [])
        new_instances[instance.index] = background_instance
    for line in codecs.open(filename, "r", "utf-8"):
        fields = line.strip().split("\t")
        index = int(fields[0])
        if index in new_instances:
            instance = new_instances[index]
            for sequence in fields[1:]:
                instance.background.append(background_class.read_from_line(sequence))
    return TextDataset(list(new_instances.values()))


class BackgroundInstance(TextInstance):
    """
    An Instance that has background knowledge associated with it.  That background knowledge can
    currently only be expressed as a list of sentences.  Maybe someday we'll expand that to allow
    other kinds of background knowledge.
    """
    def __init__(self, instance: TextInstance, background: List[TextInstance]):
        super(BackgroundInstance, self).__init__(instance.label, instance.index)
        self.instance = instance
        self.background = background

    def __str__(self):
        string = 'BackgroundInstance(\n   '
        string += str(self.instance)
        string += ',\n'
        string += '   [\n'
        for background in self.background:
            string += '      '
            string += str(background)
            string += ',\n'
        string += '   ]\n'
        string += ')'
        return string

    @overrides
    def words(self):
        words = self.instance.words()
        for background_instance in self.background:
            background_words = background_instance.words()
            for namespace in words:
                words[namespace].extend(background_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indexed_instance = self.instance.to_indexed_instance(data_indexer)
        indexed_background_instances = []
        for background_instance in self.background:
            indexed_background_instances.append(background_instance.to_indexed_instance(data_indexer))
        return IndexedBackgroundInstance(indexed_instance, indexed_background_instances)


class IndexedBackgroundInstance(IndexedInstance):
    """
    An IndexedInstance that has background knowledge associated with it, where the background
    knowledge has also been indexed.
    """
    contained_instance_type = None
    background_instance_type = None
    def __init__(self,
                 indexed_instance: IndexedInstance,
                 background_instances: List[IndexedInstance]):
        super(IndexedBackgroundInstance, self).__init__(indexed_instance.label, indexed_instance.index)
        self.indexed_instance = indexed_instance
        self.background_instances = background_instances

        # We need to set these here so that we know what kind of contained and background instances
        # we should create when we're asked for an empty IndexedBackgroundInstance or for padding.
        # Note that this assumes that you'll only ever have one type of underlying contained Instance type
        # and one type of background instance, which is a reasonable assumption given our current code.
        IndexedBackgroundInstance.contained_instance_type = indexed_instance.__class__
        if len(background_instances) > 0:
            IndexedBackgroundInstance.background_instance_type = background_instances[0].__class__

    @classmethod
    @overrides
    def empty_instance(cls):
        contained_instance = IndexedBackgroundInstance.contained_instance_type.empty_instance()
        # An empty instance contains an empty list of background instances.
        return IndexedBackgroundInstance(contained_instance, [])

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for background_indices.

        Additionally, as we currently use the same encoder for both a sentence and its background
        knowledge, we'll also modify the word_indices length to look at the background sentences
        too.
        """
        lengths = self.indexed_instance.get_padding_lengths()
        lengths['background_sentences'] = len(self.background_instances)
        # TODO(pradeep): We are using the same max sequence length for both background instances and the
        # contained instance. While this is fine when background instances are sentences, it may be inefficient
        # when they are tuples. Have a separate background sequence length. This requires making changes to
        # the lengths in most solvers.
        if len(self.background_instances) > 0:
            background_lengths = [background.get_padding_lengths() for background in self.background_instances]
            for attribute in background_lengths[0]:
                max_background_attribute_value = max(background[attribute] for background in background_lengths)
                if attribute in lengths:
                    lengths[attribute] = max(lengths[attribute], max_background_attribute_value)
                else:
                    lengths[attribute] = max_background_attribute_value
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        We let self.indexed_instance pad itself, and in this method we mostly worry about padding
        background_indices.  We need to pad it in two ways: (1) we need len(background_indices) to
        be the same for all instances, and (2) we need len(background_indices[i]) to be the same
        for all i, for all instances.  We'll use the word_indices length from the super class for
        (2).
        """
        self.indexed_instance.pad(padding_lengths)
        background_length = padding_lengths['background_sentences']

        # Padding (1): making sure we have the right number of background instances.  We also need
        # to truncate, if necessary.
        if len(self.background_instances) > background_length:
            self.background_instances = self.background_instances[:background_length]
        empty_background_instance = IndexedBackgroundInstance.background_instance_type.empty_instance()
        for _ in range(background_length - len(self.background_instances)):
            self.background_instances.append(empty_background_instance)

        # Padding (2): making sure all background instances are padded to the right length.
        for background_instance in self.background_instances:
            background_instance.pad(padding_lengths)

    @overrides
    def as_training_data(self):
        """
        This returns a complex output.  In the simplest case, the contained instance is just a
        TextClassificationInstance, with a single sentence input.  In this case, we'll return a
        tuple of (sentence_array, background_array) as the inputs (and, as always, the label from
        the contained instance).

        If the contained instance itself has multiple inputs it returns, we need the
        background_array to be second in the list (because that makes the implementation in the
        memory network solver much easier).  That means we need to change the order of things
        around a bit.
        """
        instance_inputs, label = self.indexed_instance.as_training_data()
        background_indices = []
        for background_instance in self.background_instances:
            background_inputs, _ = background_instance.as_training_data()
            if isinstance(background_inputs, tuple):
                raise RuntimeError("Received a background instance that provides multiple inputs.")
            background_indices.append(background_inputs)
        background_array = numpy.asarray(background_indices, dtype='int32')
        if isinstance(instance_inputs, tuple):
            final_inputs = (instance_inputs[0],) + (background_array,) + instance_inputs[1:]
        else:
            final_inputs = (instance_inputs, background_array)
        return final_inputs, label
