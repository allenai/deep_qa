from typing import List
import logging
import random

import tqdm

from ..common.params import Params
from ..common.util import group_by_count
from . import IndexedDataset
from .instances import IndexedInstance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DataGenerator:
    """
    A ``DataGenerator`` takes an :class:`~.dataset.IndexedDataset` and converts it into a
    generator, yielding batches suitable for training.  You might want to do this instead of just
    creating one large set of numpy arrays for a few reasons:

    #. Creating large arrays for your whole data could take a whole lot of memory, maybe more than
       is available on your machine.
    #. Creating one large array means padding all of your instances to the same length.  This
       typically means you waste a whole lot of computation on padding tokens.  Using a
       ``DataGenerator`` instead allows you to only pad each `batch` to the same length, instead of
       all of your instances across your whole dataset.  We've typically seen a 4-5x speed up just
       from doing this (partially because Keras is pretty bad at doing variable-length computation;
       the speed-up isn't quite as large with plain tensorflow, I think).
    #. If we're varying the padding lengths in each batch, we can also vary the batch size, to
       optimize GPU memory usage.  This means we'll use smaller batch sizes for big instances, and
       larger batch sizes for small instances.  We've seen speedups up to 10-12x (on top of the
       4-5x speed up above) from doing this.

    Parameters
    ----------
    text_trainer: TextTrainer
        We need access to the ``TextTrainer`` object so we can call some methods on it, such as
        :func:`~deep_qa.training.TextTrainer.get_instance_sorting_keys`.
    dynamic_padding: bool, optional (default=False)
        If ``True``, we will set padding lengths based on the data `per batch`, instead of on the
        whole dataset.  This only works if your model is structured to allow variable-length
        sequences (typically using ``None`` for specific dimensions when you build your model), and
        if you don't set padding values in
        :func:`~deep_qa.training.TextTrainer._set_padding_lengths`.  This flag specifically is read
        in :func:`~deep_qa.training.TextTrainer._set_padding_lengths` to know if we should set
        certain padding values or not.  It's handled correctly for ``num_sentence_words`` and
        ``num_word_characters`` in :class:`~deep_qa.training.TextTrainer`, but you need to be sure
        to implement it correctly in subclasses for this to work.
    adaptive_batch_sizes: bool, optional (default=False)
        Only relevant if ``dynamic_padding`` is ``True``.  If ``adaptive_batch_sizes`` is ``True``,
        we will vary the batch size to try to optimize GPU memory usage.  Because padding lengths
        are done dynamically, we can have larger batches when padding lengths are smaller,
        maximizing our usage of the GPU.  In order for this to work, you need to do two things: (1)
        override :func:`~TextTrainer._get_padding_memory_scaling` to give a big-O bound on memory
        usage given padding lengths, and (2) tune the `adaptive_memory_usage_constant` parameter
        for your particular model and GPU.  See the documentation for
        :func:`~TextTrainer._get_padding_memory_scaling` for more information.
    adaptive_memory_usage_constant: int, optional (default=None)
        Only relevant if ``adaptive_batch_sizes`` is ``True``.  This is a manually-tuned parameter,
        specific to a particular model architecture and amount of GPU memory (e.g., if you change
        the number of hidden layers in your model, this number will need to change).  See
        :func:`~TextTrainer._get_padding_memory_scaling` for more detail.
    biggest_batch_first: bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        It's only meaningful if you're using dynamic padding - this will let you try out the
        largest batch that you have in the data `first`, so that if you're going to run out of
        memory, you know it early, instead of waiting through the whole batch to find out at the
        end that you're going to crash.
    """
    def __init__(self, text_trainer, params: Params):
        self.text_trainer = text_trainer
        self.dynamic_padding = params.pop('dynamic_padding', False)
        self.adaptive_batch_sizes = params.pop('adaptive_batch_sizes', False)
        self.adaptive_memory_usage_constant = params.pop('adaptive_memory_usage_constant', False)
        self.biggest_batch_first = params.pop('biggest_batch_first', False)

        #: This field can be read after calling ``create_generator`` to get the number of steps you
        #: should take per epoch in ``model.fit_generator`` or ``model.evaluate_generator`` for
        #: this data.
        self.last_num_batches = None

    def create_generator(self, dataset: IndexedDataset):
        """
        Main external API call: converts an ``IndexedDataset`` into a data generator suitable for
        use with Keras' ``fit_generator`` and related methods.
        """
        if self.dynamic_padding:
            dataset.sort_by_padding(self.text_trainer.get_instance_sorting_keys())
        instances = dataset.instances
        if self.adaptive_batch_sizes:
            grouped_instances = self.__adaptive_grouping(instances)
        else:
            grouped_instances = group_by_count(instances, self.text_trainer.batch_size, None)
            grouped_instances[-1] = [instance for instance in grouped_instances[-1] if instance is not None]
        self.last_num_batches = len(grouped_instances)
        def generator():
            while True:
                if self.biggest_batch_first:
                    # We'll actually pop the last _two_ batches, because the last one might not
                    # be full.
                    last_batch = grouped_instances.pop()
                    penultimate_batch = grouped_instances.pop()
                    random.shuffle(grouped_instances)
                    grouped_instances.insert(0, penultimate_batch)
                    grouped_instances.insert(0, last_batch)
                else:
                    random.shuffle(grouped_instances)
                for group in grouped_instances:
                    dataset = IndexedDataset(group)
                    dataset.pad_instances(self.text_trainer.get_padding_lengths(), verbose=False)
                    yield dataset.as_training_data()
        return generator()

    def __adaptive_grouping(self, instances: List[IndexedInstance]):
        batches = []
        current_batch = []
        #max_batch_size = 60
        logger.info("Creating adatpive groups")
        for instance in tqdm.tqdm(instances):
            current_batch.append(instance)
            padding_lengths = IndexedDataset(current_batch).padding_lengths()
            big_o_memory_constant = self.text_trainer.get_padding_memory_scaling(padding_lengths)
            if len(current_batch) * big_o_memory_constant > self.adaptive_memory_usage_constant:
                current_batch.pop()
                padding_lengths = IndexedDataset(current_batch).padding_lengths()
                batches.append(current_batch)
                current_batch = [instance]
        padding_lengths = IndexedDataset(current_batch).padding_lengths()
        batches.append(current_batch)
        return batches
