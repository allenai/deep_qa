import logging
from typing import Any, Dict, List

from overrides import overrides
import numpy

from keras.layers import merge, Dense, TimeDistributed
from keras.models import Model

from ...common.checks import ConfigurationError
from ...data.dataset import TextDataset
from ...data.text_instance import SentenceCooccurrenceInstance
from ...training.pretraining.pretrainer import Pretrainer
from ..nn_solver import NNSolver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class EncoderPretrainer(Pretrainer):
    """
    Here we pretrain the encoder (including the embedding if needed) using an objective similar to word2vec.
    The input data to this pretrainer is labeled pairs of sentences with the label indicating whether the
    sentences co-occur in the same context or not. The encoded representations of the two sentences are fed to a
    MLP which is trained along with the encoder with the objective of minimizing the categorical crossentropy.
    """
    # pylint: disable=protected-access
    def __init__(self, trainer, params: Dict[str, Any]):
        if not isinstance(trainer, NNSolver):
            raise ConfigurationError("The Encoder Pretrainer needs a subclass of NNSolver")
        super(EncoderPretrainer, self).__init__(trainer, params)
        self.name = "EncoderPretrainer"
        self.loss = "binary_crossentropy"

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        return TextDataset.read_from_file(files[0], SentenceCooccurrenceInstance, tokenizer=self.trainer.tokenizer)

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool):
        # Ignoring for_train because we're pretraining.
        logger.info("Indexing encoder pretraining dataset")
        indexed_dataset = dataset.to_indexed_dataset(self.trainer.data_indexer)
        max_lengths = self.trainer._get_max_lengths()
        logger.info("Padding encoder pretraining dataset to lengths %s", str(max_lengths))
        indexed_dataset.pad_instances(max_lengths)
        inputs, labels = indexed_dataset.as_training_data()
        if isinstance(inputs[0], tuple):
            inputs = [numpy.asarray(x) for x in zip(*inputs)]
        else:
            inputs = numpy.asarray(inputs)
        return inputs, numpy.asarray(labels)

    def fit_data_indexer(self):
        dataset = self._load_dataset_from_files(self.train_files)
        self.trainer.data_indexer.fit_word_dictionary(dataset)

    @overrides
    def _build_model(self):
        sentence_shape = (self.trainer.max_sentence_length,)
        first_sentence_input, embedded_first_sentence = self.trainer._get_embedded_sentence_input(sentence_shape,
                                                                                                  "first_sentence")
        second_sentence_input, embedded_second_sentence = self.trainer._get_embedded_sentence_input(
                sentence_shape, "second_sentence")
        sentence_encoder = self.trainer._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        encoded_first_sentence = sentence_encoder(embedded_first_sentence)
        encoded_second_sentence = sentence_encoder(embedded_second_sentence)
        merged_sentences = merge([encoded_first_sentence, encoded_second_sentence], mode='concat')
        # The activation is a sigmoid because the label is one integer instead of a one-hot array, and it
        # does not make sense to do a softmax on one value.
        binary_prediction = Dense(1, activation='sigmoid', name='context_prediction')(merged_sentences)
        return Model(input=[first_sentence_input, second_sentence_input], output=binary_prediction)
