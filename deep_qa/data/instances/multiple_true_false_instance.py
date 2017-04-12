from collections import defaultdict
from typing import Dict, List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer


class MultipleTrueFalseInstance(TextInstance):
    """
    A MultipleTrueFalseInstance is a grouping of other Instances, where exactly one of those Instances
    must have label True.  This means that this really needs to be backed by TrueFalseInstances,
    though those could have already been wrapped in BackgroundInstances.

    When this is converted to training data, it will group all of those option Instances into a
    single training instance, with a label that is an index to the answer option that is correct
    for its label.
    """
    def __init__(self, options: List[TextInstance]):
        self.options = options
        no_label = len(list([i for i in options if i.label is not None])) == 0
        if no_label:
            label = None
        else:
            positive_index = [index for index, instance in enumerate(options) if instance.label is True]
            assert len(positive_index) == 1
            label = positive_index[0]
        super(MultipleTrueFalseInstance, self).__init__(label, None)

    def __str__(self):
        options_string = ',\n    '.join([str(x) for x in self.options])
        return 'MultipleTrueFalseInstance(  \n(\n    ' + options_string + '\n  ),\n  ' + \
                str(self.label) + '\n)'

    @overrides
    def words(self):
        words = defaultdict(list)
        for option in self.options:
            option_words = option.words()
            for namespace in option_words:
                words[namespace].extend(option_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        indexed_options = [option.to_indexed_instance(data_indexer) for option in self.options]
        return IndexedMultipleTrueFalseInstance(indexed_options, self.label)


class IndexedMultipleTrueFalseInstance(IndexedInstance):
    """
    A MultipleTrueFalseInstance that has been indexed.  MultipleTrueFalseInstance has a better
    description of what this represents.
    """
    def __init__(self, options: List[IndexedInstance], label):
        super(IndexedMultipleTrueFalseInstance, self).__init__(label=label, index=None)
        self.options = options

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedMultipleTrueFalseInstance([], None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Here we return the max of get_lengths on all of the Instances in self.options.
        """
        padding_lengths = {}
        padding_lengths['num_options'] = len(self.options)
        lengths = [instance.get_lengths() for instance in self.options]
        if not lengths:
            return padding_lengths
        for key in lengths[0]:
            padding_lengths[key] = max(x[key] for x in lengths)
        return padding_lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        This method pads all of the underlying Instances in self.options.
        """
        num_options = padding_lengths['num_options']

        # First we pad the number of options.
        while len(self.options) < num_options:
            self.options.append(self.options[0].empty_instance())
        self.options = self.options[:num_options]

        # Then we pad each option.
        for instance in self.options:  # type: IndexedInstance
            instance.pad(padding_lengths)

    @overrides
    def as_training_data(self):
        inputs = []
        unzip_inputs = False
        for option in self.options:
            option_input, _ = option.as_training_data()
            if isinstance(option_input, tuple):
                unzip_inputs = True
            inputs.append(option_input)
        if unzip_inputs:
            inputs = tuple(zip(*inputs))  # pylint: disable=redefined-variable-type
            inputs = tuple([numpy.asarray(x) for x in inputs])
        else:
            inputs = numpy.asarray(inputs)
        if self.label is None:
            label = None
        else:
            label = numpy.zeros(len(self.options))
            label[self.label] = 1
        return inputs, label
