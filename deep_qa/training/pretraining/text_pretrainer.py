import logging
from typing import Any, Dict, List
from overrides import overrides

import numpy

from .pretrainer import Pretrainer
from ..text_trainer import TextTrainer
from ...common.checks import ConfigurationError
from ...data.instances.instance import Instance
from ...data.dataset import TextDataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextPretrainer(Pretrainer):
    # pylint: disable=abstract-method,protected-access
    """
    A TextPretrainer is a Pretrainer that has a TextTrainer as the model that it is doing
    pre-training for.  Just as TextTrainer adds helper methods to Trainer for dealing with word
    sequences, this adds very similar helper methods to Pretrainer for dealing with word sequences.
    """
    def __init__(self, trainer: TextTrainer, params: Dict[str, Any]):
        if not isinstance(trainer, TextTrainer):
            raise ConfigurationError("TextPretrainers need a subclass of TextTrainer")
        super(TextPretrainer, self).__init__(trainer, params)

    def _instance_type(self) -> Instance:
        """
        When reading datasets, what instance type should we create?  We could set this to
        self.trainer._instance_type() by default, but that isn't always what you want to do.  So
        instead of giving a default here, we'll make this an abstract method, so you're forced to
        think about this and give the right Instance type.
        """
        raise NotImplementedError

    def _load_dataset_from_files(self, files: List[str]):
        """
        This method assumes you have a TextDataset that can be read from a single file.  If you
        have something more complicated, you'll need to override this method (possibly calling this
        to process the first file, and processing other files in the subclass).
        """
        return TextDataset.read_from_file(files[0], self._instance_type())

    def fit_data_indexer(self):
        """
        This method allows the pre-training data to add words to the TextTrainer's vocabulary.
        """
        dataset = self._load_dataset_from_files(self.train_files)
        self.trainer.data_indexer.fit_word_dictionary(dataset)

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool,
                      update_data_indexer=True):
        """
        This does basically the same thing as TextTrainer._prepare_data(), except for the things
        done when for_train is True.  We also rely on our contained Trainer instance for some of
        the variables in here, where TextTrainer relies on `self`.

        Specifically, what we do here is convert a TextDataset to an IndexedDataset, using the
        Trainer's DataIndexer, then do padding on the indexed data (using the max lengths set by
        the Trainer), and convert it to actual training arrays that can be used with Keras.  What
        those training arrays look like is determined by the type of Instances that you're using,
        which is determined by self._instance_type(), just as in TextDataset.

        How exactly you want to handle the pre-training data affecting the max_lengths of the model
        is a complicated question.  We handle that by processing the training data first, having
        the Trainer set its max_lengths, and then using those for padding here.  If you want the
        pre-training data to set max_lengths instead of the actual training data, you'll have to
        change Trainer.train().
        """
        logger.info("Indexing pretraining dataset")
        indexed_dataset = dataset.to_indexed_dataset(self.trainer.data_indexer)
        max_lengths = self.trainer._get_max_lengths()
        logger.info("Padding pretraining dataset to lengths %s", str(max_lengths))
        indexed_dataset.pad_instances(max_lengths)
        inputs, labels = indexed_dataset.as_training_data()
        if isinstance(inputs[0], tuple):
            inputs = [numpy.asarray(x) for x in zip(*inputs)]
        else:
            inputs = numpy.asarray(inputs)
        return inputs, numpy.asarray(labels)
