import codecs
import random

from typing import List

from .instance import BackgroundInstance, Instance, IndexedInstance
from .index_data import DataIndexer

class Dataset(object):
    """
    A collection of Instances, with some helper methods.

    I was originally thinking I'd have several kinds of Dataset subclasses, but in the end decided
    that subclassing Instance made more sense.
    """
    def __init__(self, instances: List[Instance]):
        """
        A Dataset just takes a list of instances in its constructor.  It's important that all
        subclasses have an identical constructor to this (though possibly with different Instance
        types).  If you change the constructor, you also have to override all methods in this base
        class that call the constructor, such as `merge()` and `truncate()`.
        """
        self.instances = instances

    def to_indexed_dataset(self, data_indexer: DataIndexer) -> 'IndexedDataset':
        '''
        Converts the Dataset into an IndexedDataset, given a DataIndexer.
        '''
        indexed_instances = [instance.to_indexed_instance(data_indexer) for instance in self.instances]
        return IndexedDataset(indexed_instances)

    def merge(self, other: 'Dataset') -> 'Dataset':
        """
        Combine two datasets.  If you call try to merge two Datasets of the same subtype, you will
        end up with a Dataset of the same type (i.e., calling IndexedDataset.merge() with another
        IndexedDataset will return an IndexedDataset).  If the types differ, this method currently
        raises an error, because the underlying Instance objects are not currently type compatible.
        """
        if type(self) is type(other):
            return self.__class__(self.instances + other.instances)
        else:
            raise RuntimeError("Cannot merge datasets with different types")

    def truncate(self, max_instances: int):
        """
        If there are more instances than `max_instances` in this dataset, returns a new dataset
        with a random subset of size `max_instances`.  If there are fewer than `max_instances`
        already, we just return self.
        """
        if len(self.instances) <= max_instances:
            return self
        new_instances = [i for i in self.instances]
        random.shuffle(new_instances)
        return self.__class__(new_instances[:max_instances])

    @staticmethod
    def read_from_file(filename: str, label: bool=None):
        lines = [x.strip() for x in codecs.open(filename, "r", "utf-8").readlines()]
        instances = [Instance.read_from_line(x, label) for x in lines]
        return Dataset(instances)

    @staticmethod
    def read_background_from_file(dataset: 'Dataset', filename: str) -> 'Dataset':
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
        new_instances = {}
        for instance in dataset.instances:
            background_instance = BackgroundInstance(instance.text, [], instance.label, instance.index)
            new_instances[instance.index] = background_instance
        for line in codecs.open(filename, "r", "utf-8"):
            fields = line.strip().split("\t")
            index = int(fields[0])
            if index in new_instances:
                instance = new_instances[index]
                for sequence in fields[1:]:
                    instance.background.append(sequence)
        return Dataset(list(new_instances.values()))


class IndexedDataset(object):
    """
    A collection of IndexedInstances, with some helper methods.
    """
    def __init__(self, instances: List[IndexedInstance]):
        self.instances = instances

    def max_lengths(self):
        lengths = [instance.get_lengths() for instance in self.instances]
        return [max(dimension_lengths) for dimension_lengths in zip(*lengths)]

    def pad_instances(self, max_lengths: List[int]):
        """
        Make all of the IndexedInstances in the dataset have the same length by padding them (in
        the front) with zeros.

        If max_length is given for a particular dimension, we will pad all instances to that length
        (including left-truncating instances if necessary).  If not, we will find the longest
        instance and pad all instances to that length.  Note that max_lengths is a _List_, not an
        int - there could be several dimensions on which we need to pad, depending on what kind of
        instance we are dealing with.

        This method _modifies_ the current object, it does not return a new IndexedDataset.
        """
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular dimension.  If we were, we use that instead of the
        # instance-based one.
        instance_max_lengths = self.max_lengths()
        zipped_maxes = zip(max_lengths, instance_max_lengths)
        max_lengths = []
        for given, from_instance in zipped_maxes:
            if given is not None:
                max_lengths.append(given)
            else:
                max_lengths.append(from_instance)

        for instance in self.instances:
            instance.pad(max_lengths)

    def as_training_data(self, shuffle=True):
        """
        Takes each IndexedInstance and converts it into (inputs, labels), according to the
        Instance's as_training_data() method.  Note that you might need to call numpy.asarray() on
        the results of this; we don't do that for you, because the inputs might be complicated.
        """
        inputs = []
        labels = []
        instances = self.instances
        if shuffle:
            random.shuffle(instances)
        for instance in instances:
            instance_inputs, label = instance.as_training_data()
            inputs.append(instance_inputs)
            labels.append(label)
        return inputs, labels
