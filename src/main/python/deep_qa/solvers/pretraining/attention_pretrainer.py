import logging
from typing import Any, Dict, List

from overrides import overrides
import numpy

from keras import backend as K
from keras.layers import merge, Dropout, TimeDistributed
from keras.models import Model

from ...training.pretraining.pretrainer import Pretrainer
from ...data.dataset import TextDataset

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AttentionPretrainer(Pretrainer):
    # While it's not great, we need access to a few of the internals of the trainer, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, trainer, params: Dict[str, Any]):
        self.train_file = params.pop('train_file')
        self.background_file = params.pop('background_file')
        super(AttentionPretrainer, self).__init__(trainer, params)

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = TextDataset.read_from_file(files[0],
                                             self.trainer._instance_type(),
                                             tokenizer=self.trainer.tokenizer)
        return TextDataset.read_labeled_background_from_file(dataset, files[1])

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool):
        # We ignore the for_train parameter here, because we're not training, we're pre-training.
        # So we basically do the same thing as NNSolver._prepare_data(), except for fitting a data
        # indexer (which already happened), and setting max_lengths.
        # TODO(matt): might be worth making a TextPretrainer whenever we make a TextTrainer, to
        # share some of this common data preparation code.
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

    def fit_data_indexer(self):
        dataset = self._load_dataset_from_files(self._get_training_files())
        self.trainer.data_indexer.fit_word_dictionary(dataset)

    @overrides
    def _get_training_files(self):
        return [self.train_file, self.background_file]

    @overrides
    def _build_model(self):
        """
        This model basically just pulls out the first half of the memory network model, up until
        the first attention layer.

        Because the trainer we're pretraining might have some funny input shapes, we don't use
        trainer._get_question_shape() directly; instead we re-create it for the case where we don't
        have TimeDistributed input.
        """
        # What follows is a lightly-edited version of the code from
        # MemoryNetworkSolver._build_model().
        sentence_shape = (self.trainer.max_sentence_length,)
        background_shape = (self.trainer.max_knowledge_length, self.trainer.max_sentence_length)

        sentence_input_layer, sentence_embedding = self.trainer._get_embedded_sentence_input(
                input_shape=sentence_shape, name_prefix="sentence")
        background_input_layer, background_embedding = self.trainer._get_embedded_sentence_input(
                input_shape=background_shape, name_prefix="background")

        sentence_encoder = self.trainer._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer

        background_encoder = TimeDistributed(sentence_encoder, name='background_encoder')
        encoded_sentence = sentence_encoder(sentence_embedding)  # (samples, word_dim)
        encoded_background = background_encoder(background_embedding)  # (samples, background_len, word_dim)

        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=1),
                                                       layer_outs[1]],
                                                      axis=1)
        merged_encoded_rep = merge([encoded_sentence, encoded_background],
                                   mode=merge_mode,
                                   output_shape=(self.trainer.max_knowledge_length + 1,
                                                 self.trainer.max_sentence_length),
                                   name='concat_sentence_with_background_%d' % 0)

        regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
        knowledge_selector = self.trainer._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(regularized_merged_rep)

        input_layers = [sentence_input_layer, background_input_layer]
        return Model(input=input_layers, output=attention_weights)
