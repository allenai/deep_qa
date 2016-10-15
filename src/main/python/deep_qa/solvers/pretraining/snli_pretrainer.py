import logging
from typing import Any, Dict, List

from overrides import overrides
import numpy

from keras import backend as K
from keras.layers import merge, Lambda, TimeDistributed
from keras.models import Model

from ...common.checks import ConfigurationError
from ...data.dataset import TextDataset
from ...data.text_instance import SnliInstance
from ...training.pretraining.pretrainer import Pretrainer
from ..nn_solver import NNSolver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SnliPretrainer(Pretrainer):
    # pylint: disable=abstract-method
    """
    An SNLI pretrainer is a Pretrainer that uses the Stanford Natural Language Inference dataset in
    some way.  This is still an abstract class; the only thing we do is add a load_data() method
    for easily getting SNLI inputs.
    """
    def __init__(self, trainer, params: Dict[str, Any]):
        if not isinstance(trainer, NNSolver):
            raise ConfigurationError("The SNLI Pretrainer needs a subclass of NNSolver")
        super(SnliPretrainer, self).__init__(trainer, params)
        self.name = "SnliPretrainer"

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        return TextDataset.read_from_file(files[0], SnliInstance, tokenizer=self.trainer.tokenizer)

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool):
        # We ignore the for_train parameter here, because we're not training, we're pre-training.
        # So we basically do the same thing as NNSolver._prepare_data(), except for fitting a data
        # indexer (which already happened), and setting max_lengths.

        # While it's not great, we need access to a few of the internals of the trainer, so we'll
        # disable protected access checks.  pylint: disable=protected-access
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
        dataset = self._load_dataset_from_files(self.train_files)
        self.trainer.data_indexer.fit_word_dictionary(dataset)


class SnliEntailmentPretrainer(SnliPretrainer):
    """
    This pretrainer uses SNLI data to train the encoder and entailment portions of the model.  We
    construct a simple model that uses the text and hypothesis and input, passes them through the
    sentence encoder and then the entailment layer, and predicts the SNLI label (entails,
    contradicts, neutral).
    """
    # While it's not great, we need access to a few of the internals of the trainer, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, trainer, params: Dict[str, any]):
        super(SnliEntailmentPretrainer, self).__init__(trainer, params)
        self.name = "SnliEntailmentPretrainer"
        if self.trainer.has_sigmoid_entailment:
            self.loss = 'binary_crossentropy'

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = super(SnliEntailmentPretrainer, self)._load_dataset_from_files(files)
        # The label that we get is always true/false, but some solvers need this encoded as a
        # single output dimension, and some need it encoded as two.  So, when we create our
        # dataset, we need to know which kind of label to output.
        if self.trainer.has_sigmoid_entailment:
            score_activation = 'sigmoid'
        else:
            score_activation = 'softmax'
        instances = [x.to_entails_instance(score_activation) for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _build_model(self):
        sentence_shape = (self.trainer.max_sentence_length,)
        text_input, embedded_text = self.trainer._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.trainer._get_embedded_sentence_input(sentence_shape,
                                                                                          "hypothesis")
        sentence_encoder = self.trainer._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)
        combine_layer = Lambda(self._combine_inputs,
                               output_shape=lambda x: (x[0][0], x[0][1]*3),
                               name='concat_entailment_inputs')
        entailment_input = combine_layer([hypothesis_encoding, text_encoding])
        entailment_combiner = self.trainer._get_entailment_input_combiner()
        while isinstance(entailment_combiner, TimeDistributed):
            entailment_combiner = entailment_combiner.layer
        combined_input = entailment_combiner(entailment_input)
        entailment_model = self.trainer._get_entailment_model()
        hidden_input = combined_input
        for layer in entailment_model.hidden_layers:
            hidden_input = layer(hidden_input)
        entailment_score = entailment_model.score_layer(hidden_input)
        return Model(input=[text_input, hypothesis_input], output=entailment_score)

    @staticmethod
    def _combine_inputs(inputs):
        hypothesis, text = inputs
        empty_background = K.zeros_like(hypothesis)
        entailment_input = K.concatenate([hypothesis, text, empty_background], axis=1)
        return entailment_input


class SnliAttentionPretrainer(SnliPretrainer):
    """
    This pretrainer uses SNLI data to train the attention component of the model.  Because the
    attention component doesn't have a whole lot of parameters (none in some cases), this is
    largely training the encoder.

    The way we do this is by converting the typical entailment decision into a binary decision
    (relevant / not relevant, where entails and contradicts are both considered relevant, while
    neutral is not), and training the attention model to predict the binary label.

    To keep things easy, we'll construct the data as if the text is a "background" of length 1 in
    the memory network, using the same fancy concatenation seen in the memory network trainer.

    Note that this will only train the first knowledge selector.  We should probably re-use the
    layers, though, actually...  Pradeep: shouldn't we be doing that?  Using the same layers for
    the knowledge selector and the memory updater at each memory step?
    """
    # While it's not great, we need access to a few of the internals of the trainer, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, trainer, params: Dict[str, any]):
        super(SnliAttentionPretrainer, self).__init__(trainer, params)
        self.name = "SnliAttentionPretrainer"
        self.loss = 'binary_crossentropy'

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = super(SnliAttentionPretrainer, self)._load_dataset_from_files(files)
        instances = [x.to_attention_instance() for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _build_model(self):
        sentence_shape = (self.trainer.max_sentence_length,)
        text_input, embedded_text = self.trainer._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.trainer._get_embedded_sentence_input(sentence_shape,
                                                                                          "hypothesis")
        sentence_encoder = self.trainer._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)

        merge_mode = lambda x: K.concatenate([K.expand_dims(x[0], dim=1), K.expand_dims(x[1], dim=1)],
                                             axis=1)
        merged_encoded_rep = merge([hypothesis_encoding, text_encoding],
                                   mode=merge_mode,
                                   output_shape=(2, self.trainer.embedding_size),
                                   name='concat_hypothesis_with_text')
        knowledge_selector = self.trainer._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(merged_encoded_rep)
        input_layers = [text_input, hypothesis_input]
        return Model(input=input_layers, output=attention_weights)
