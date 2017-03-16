import logging
from typing import Any, Dict, List

from overrides import overrides

from keras import backend as K
from keras.layers import merge, Input, Lambda, TimeDistributed

from ....data.dataset import TextDataset
from ....data.instances.snli_instance import SnliInstance
from ....training.pretraining.text_pretrainer import TextPretrainer
from ....training.models import DeepQaModel

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class SnliEntailmentPretrainer(TextPretrainer):
    """
    This pretrainer uses SNLI data to train the encoder and entailment portions of the model.  We
    construct a simple model that uses the text and hypothesis and input, passes them through the
    sentence encoder and then the entailment layer, and predicts the SNLI label (entails,
    contradicts, neutral).
    """
    def __init__(self, trainer, params: Dict[str, Any]):
        super(SnliEntailmentPretrainer, self).__init__(trainer, params)
        self.name = "SnliEntailmentPretrainer"
        if self.trainer.has_sigmoid_entailment:
            self.loss = 'binary_crossentropy'

    @overrides
    def _instance_type(self):
        return SnliInstance

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
        # pylint: disable=protected-access
        sentence_shape = self.trainer._get_sentence_shape()
        text_input = Input(shape=sentence_shape, dtype='int32', name="text_input")
        hypothesis_input = Input(shape=sentence_shape, dtype='int32', name="hypothesis_input")
        embedded_text = self.trainer._embed_input(text_input)
        embedded_hypothesis = self.trainer._embed_input(hypothesis_input)

        sentence_encoder = self.trainer._get_encoder()
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
        return DeepQaModel(input=[text_input, hypothesis_input], output=entailment_score)

    @staticmethod
    def _combine_inputs(inputs):
        hypothesis, text = inputs
        empty_background = K.zeros_like(hypothesis)
        entailment_input = K.concatenate([hypothesis, text, empty_background], axis=1)
        return entailment_input


class SnliAttentionPretrainer(TextPretrainer):
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
    def __init__(self, trainer, params: Dict[str, Any]):
        super(SnliAttentionPretrainer, self).__init__(trainer, params)
        self.name = "SnliAttentionPretrainer"
        self.loss = 'binary_crossentropy'

    @overrides
    def _instance_type(self):
        return SnliInstance

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = super(SnliAttentionPretrainer, self)._load_dataset_from_files(files)
        instances = [x.to_attention_instance() for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _build_model(self):
        # pylint: disable=protected-access
        sentence_shape = self.trainer._get_sentence_shape()
        text_input = Input(shape=sentence_shape, dtype='int32', name="text_input")
        hypothesis_input = Input(shape=sentence_shape, dtype='int32', name="hypothesis_input")
        embedded_text = self.trainer._embed_input(text_input)
        embedded_hypothesis = self.trainer._embed_input(hypothesis_input)

        sentence_encoder = self.trainer._get_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)

        merge_mode = lambda x: K.concatenate([K.expand_dims(x[0], dim=1),
                                              K.expand_dims(x[1], dim=1),
                                              K.expand_dims(x[2], dim=1)],
                                             axis=1)
        merged_encoded_rep = merge([hypothesis_encoding, hypothesis_encoding, text_encoding],
                                   mode=merge_mode,
                                   output_shape=(3, self.trainer.embedding_dim['words']),
                                   name='concat_hypothesis_with_text')
        knowledge_selector = self.trainer._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(merged_encoded_rep)
        input_layers = [text_input, hypothesis_input]
        return DeepQaModel(input=input_layers, output=attention_weights)
