from typing import List

import numpy
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from .background_instance import BackgroundInstance, IndexedBackgroundInstance
from ..data_indexer import DataIndexer


class LabeledBackgroundInstance(BackgroundInstance):
    """
    An Instance that has background knowledge associated with it, where the label is **attention
    over the background knowledge**.  This is basically identical to BackgroundInstance, except for
    the label.  In that case, the label is taken from the wrapped instance.  In this case, the
    label is over the background knowledge itself.

    This kind of instance is used for pre-training the attention component of a memory network (or
    perhaps for training an answer sentence selection model, though we don't currently have any of
    those models).

    The label is a List[int], containing the indices of all positive background sentences (e.g.,
    [5, 8], or [2], etc.).
    """
    def __init__(self, instance: TextInstance, background: List[str], label: List[int]):
        super(LabeledBackgroundInstance, self).__init__(instance, background)
        self.label = label

    def __str__(self):
        return 'LabeledBackgroundInstance(' + str(self.instance) + ', ' + str(self.background) + \
                ', ' + str(self.label) + ')'

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        instance = super(LabeledBackgroundInstance, self).to_indexed_instance(data_indexer)
        return IndexedLabeledBackgroundInstance(instance.indexed_instance,
                                                instance.background_instances,
                                                self.label)


class IndexedLabeledBackgroundInstance(IndexedBackgroundInstance):
    """
    This is an IndexedBackgroundInstance that has a different label.  Instead of passing through
    the contained instance's label, we have labeled attention over the background data.  So this
    object behaves identically to IndexedBackgroundInstance in everything except the label.

    See text_instance.LabeledBackgroundInstance for a little more detail.
    """
    def __init__(self,
                 indexed_instance: IndexedInstance,
                 background_indices: List[List[int]],
                 label: List[int]):
        super(IndexedLabeledBackgroundInstance, self).__init__(indexed_instance, background_indices)
        self.label = label

    @overrides
    def as_training_data(self):
        """
        All we do here is overwrite the label from IndexedBackgroundInstance.
        """
        inputs, _ = super(IndexedLabeledBackgroundInstance, self).as_training_data()
        if self.label is None:
            label = None
        else:
            label = numpy.zeros(len(self.background_instances))
            for index in self.label:
                label[index] = 1
        return inputs, label
