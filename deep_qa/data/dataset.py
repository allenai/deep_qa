import codecs
from collections import OrderedDict
import itertools
import logging
from typing import Dict, List

import tqdm

from .instances.instance import Instance, TextInstance, IndexedInstance
from .instances.background_instance import BackgroundInstance
from .instances.labeled_background_instance import LabeledBackgroundInstance
from .instances.multiple_true_false_instance import MultipleTrueFalseInstance
from .data_indexer import DataIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Dataset:
    """
    A collection of Instances.

    This base class has general methods that apply to all collections of Instances.  That basically
    is just methods that operate on sets, like merging and truncating.
    """
    def __init__(self, instances: List[Instance]):
        """
        A Dataset just takes a list of instances in its constructor.  It's important that all
        subclasses have an identical constructor to this (though possibly with different Instance
        types).  If you change the constructor, you also have to override all methods in this base
        class that call the constructor, such as `merge()` and `truncate()`.
        """
        self.instances = instances

    def can_be_converted_to_multiple_choice(self):
        """
        This method checks that dataset matches the assumptions we make about question data: that
        it is a list of sentences corresponding to four-choice questions, with one correct answer
        for every four instances.

        So, specifically, we check that the number of instances is a multiple of four, and we check
        that each group of four instances has exactly one instance with label True, and all other
        labels are False (i.e., no None labels for validation data).
        """
        # TODO(matt): this method does not belong on `Dataset`.
        for instance in self.instances:
            if isinstance(instance, MultipleTrueFalseInstance):
                return False
        if len(self.instances) % 4 != 0:
            return False
        questions = zip(*[self.instances[i::4] for i in range(4)])
        for question in questions:
            question_labels = [instance.label for instance in question]
            label_counts = {x: question_labels.count(x) for x in set(question_labels)}
            if label_counts[True] != 1:
                return False
            if label_counts[False] != 3:
                return False
        return True

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
        return self.__class__(new_instances[:max_instances])


class TextDataset(Dataset):
    """
    A Dataset of TextInstances, with a few helper methods.

    TextInstances aren't useful for much with Keras until they've been indexed.  So this class just
    has methods to read in data from a file and converting it into other kinds of Datasets.
    """
    def __init__(self, instances: List[TextInstance]):
        super(TextDataset, self).__init__(instances)

    def to_indexed_dataset(self, data_indexer: DataIndexer) -> 'IndexedDataset':
        '''
        Converts the Dataset into an IndexedDataset, given a DataIndexer.
        '''
        indexed_instances = [instance.to_indexed_instance(data_indexer) for instance in tqdm.tqdm(self.instances)]
        return IndexedDataset(indexed_instances)

    def to_question_dataset(self) -> 'Dataset':
        # TODO(matt): this method does not belong on `TextDataset`
        assert self.can_be_converted_to_multiple_choice()
        questions = zip(*[self.instances[i::4] for i in range(4)])
        question_instances = []
        for question in questions:
            question_instances.append(MultipleTrueFalseInstance(question))
        return TextDataset(question_instances)

    @staticmethod
    def read_from_file(filename: str, instance_class, label: bool=None):
        lines = [x.strip() for x in tqdm.tqdm(codecs.open(filename, "r",
                                                          "utf-8").readlines())]
        return TextDataset.read_from_lines(lines, instance_class, label)

    @staticmethod
    def read_from_lines(lines: List[str], instance_class, label: bool=None):
        instances = [instance_class.read_from_line(x, label) for x in lines]
        labels = [(x.label, x) for x in instances]
        labels.sort(key=lambda x: str(x[0]))
        label_counts = [(label, len([x for x in group]))
                        for label, group in itertools.groupby(labels, lambda x: x[0])]
        label_count_str = str(label_counts)
        if len(label_count_str) > 100:
            label_count_str = label_count_str[:100] + '...'
        logger.info("Finished reading dataset; label counts: %s", label_count_str)
        return TextDataset(instances)

    @staticmethod
    def read_background_from_file(dataset: 'TextDataset', filename: str, background_class):
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
                    instance.background.append(background_class.read_from_line(sequence, None))
        return TextDataset(list(new_instances.values()))

    @staticmethod
    def read_labeled_background_from_file(dataset: 'TextDataset', filename: str) -> 'TextDataset':
        """
        Reads a file formatted as labeled background information and matches the background to the
        sentences in the given dataset.  The given dataset must have instance indices, so we can
        match the background information in the file to the instances in the dataset.

        This is like read_background_from_file(), except we create LabeledBackgroundInstances
        instead of BackgroundInstances.

        The format for the file is assumed to be the following:
        [sentence index][tab][correct background indices][tab][background 1][tab][background 2][tab][...]
        where [sentence index] corresponds to the index of one of the instances in `dataset`, and
        [correct background indices] is a comma-separated list of (0-indexed) integers, pointing to
        the background sentences which are positive examples.
        """
        new_instances = {}
        for instance in dataset.instances:
            background_instance = LabeledBackgroundInstance(instance, [], [])
            new_instances[instance.index] = background_instance
        for line in codecs.open(filename, "r", "utf-8"):
            fields = line.strip().split("\t")
            index = int(fields[0])
            correct_background_indices = [int(x) for x in fields[1].split(',')]
            if index in new_instances:
                instance = new_instances[index]
                instance.label = correct_background_indices
                for sequence in fields[2:]:
                    instance.background.append(sequence)
        return TextDataset(list(new_instances.values()))


class IndexedDataset(Dataset):
    """
    A Dataset of IndexedInstances, with some helper methods.

    IndexedInstances have text sequences replaced with lists of word indices, and are thus able to
    be padded to consistent lengths and converted to training inputs.
    """
    def __init__(self, instances: List[IndexedInstance]):
        super(IndexedDataset, self).__init__(instances)

    def padding_lengths(self):
        padding_lengths = {}
        lengths = [instance.get_lengths() for instance in self.instances]
        if not lengths:
            return padding_lengths
        for key in lengths[0]:
            padding_lengths[key] = max(x[key] if key in x else 0 for x in lengths)
        return padding_lengths

    def pad_instances(self, padding_lengths: Dict[str, int]=None, verbose: bool=True):
        """
        Makes all of the ``IndexedInstances`` in the dataset have the same length by padding them.
        This ``Dataset`` object doesn't know what things there are in the ``Instance`` to pad, but
        the ``Instances`` do, and so does the model that called us, passing in a
        ``padding_lengths`` dictionary.  The keys in that dictionary must match the lengths that
        the ``Instance`` knows about.

        Given that, this method does two things: (1) it asks each of the ``Instances`` what their
        padding lengths are, and takes a max (using :func:`~IndexedDataset.padding_lengths()`).  It
        then reconciles those values with the ``padding_lengths`` we were passed as an argument to
        this method, and pads the instances with :func:`IndexedInstance.pad()`.  If
        ``padding_lengths`` has a particular key specified with a value, that value takes
        precedence over whatever we computed in our data.  TODO(matt): with dynamic padding, we
        should probably have this be a max padding length, not a hard setting, but that requires
        some API changes.

        This method `modifies` the current object, it does not return a new ``IndexedDataset``.

        Parameters
        ----------
        padding_lengths: Dict[str, int]
            If a key is present in this dictionary with a non-`None` value, we will pad to that
            length instead of the length calculated from the data.  This lets you, e.g., set a
            maximum value for sentence length, or word length, if you want to throw out long
            sequences.
        verbose: bool, optional (default=True)
            Should we output logging information when we're doing this padding?  If the dataset is
            large, this is nice to have, because padding a large dataset could take a long time.
            But if you're doing this inside of a data generator, having all of this output per
            batch is a bit obnoxious.
        """
        # First we need to decide _how much_ to pad.  To do that, we find the max length for all
        # relevant padding decisions from the instances themselves.  Then we check whether we were
        # given a max length for a particular dimension.  If we were, we use that instead of the
        # instance-based one.
        if verbose:
            logger.info("Getting max lengths from instances")
        instance_padding_lengths = self.padding_lengths()
        if verbose:
            logger.info("Instance max lengths: %s", str(instance_padding_lengths))
        lengths_to_use = {}
        for key in instance_padding_lengths:
            if padding_lengths and padding_lengths[key] is not None:
                lengths_to_use[key] = padding_lengths[key]
            else:
                lengths_to_use[key] = instance_padding_lengths[key]
        if verbose:
            logger.info("Now actually padding instances to length: %s", str(lengths_to_use))
            for instance in tqdm.tqdm(self.instances):
                instance.pad(lengths_to_use)
        else:
            for instance in self.instances:
                instance.pad(lengths_to_use)

    def as_training_data(self):
        """
        Takes each IndexedInstance and converts it into (inputs, labels), according to the
        Instance's as_training_data() method.  Note that you might need to call numpy.asarray() on
        the results of this; we don't do that for you, because the inputs might be complicated.
        """
        inputs = []
        labels = []
        instances = self.instances
        for instance in instances:
            instance_inputs, label = instance.as_training_data()
            inputs.append(instance_inputs)
            labels.append(label)
        return inputs, labels
