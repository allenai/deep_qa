from overrides import overrides

from keras import backend as K
from keras.layers import merge, Lambda, TimeDistributed
from keras.models import Model

from .pretrainer import Pretrainer
from ...data.dataset import TextDataset
from ...data.text_instance import SnliInstance

class SnliPretrainer(Pretrainer):
    # pylint: disable=abstract-method
    """
    An SNLI pretrainer is a Pretrainer that uses the Stanford Natural Language Inference dataset in
    some way.  This is still an abstract class; the only thing we do is add a load_data() method
    for easily getting SNLI inputs.
    """
    def __init__(self, solver, snli_file, **kwargs):
        super(SnliPretrainer, self).__init__(solver, **kwargs)
        self.snli_file = snli_file

    @overrides
    def _load_dataset(self):
        return TextDataset.read_from_file(self.snli_file, SnliInstance, self.solver.tokenizer)


class SnliEntailmentPretrainer(SnliPretrainer):
    """
    This pretrainer uses SNLI data to train the encoder and entailment portions of the model.  We
    construct a simple model that uses the text and hypothesis and input, passes them through the
    sentence encoder and then the entailment layer, and predicts the SNLI label (entails,
    contradicts, neutral).
    """
    # While it's not great, we need access to a few of the internals of the solver, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, solver, snli_file, **kwargs):
        super(SnliEntailmentPretrainer, self).__init__(solver, snli_file, **kwargs)
        if self.solver.has_binary_entailment:
            self.loss = 'binary_crossentropy'

    @overrides
    def _load_dataset(self):
        dataset = super(SnliEntailmentPretrainer, self)._load_dataset()
        # The label that we get is always true/false, but some solvers need this encoded as a
        # single output dimension, and some need it encoded as two.  So, when we create our
        # dataset, we need to know which kind of label to output.
        if self.solver.has_binary_entailment:
            score_activation = 'sigmoid'
        else:
            score_activation = 'softmax'
        instances = [x.to_entails_instance(score_activation) for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _get_model(self):
        sentence_shape = (self.solver.max_sentence_length,)
        text_input, embedded_text = self.solver._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.solver._get_embedded_sentence_input(sentence_shape,
                                                                                         "hypothesis")
        sentence_encoder = self.solver._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)
        combine_layer = Lambda(self._combine_inputs,
                               output_shape=lambda x: (x[0][0], x[0][1]*3),
                               name='concat_entailment_inputs')
        entailment_input = combine_layer([hypothesis_encoding, text_encoding])
        entailment_combiner = self.solver._get_entailment_combiner()
        while isinstance(entailment_combiner, TimeDistributed):
            entailment_combiner = entailment_combiner.layer
        combined_input = entailment_combiner(entailment_input)
        entailment_model = self.solver.entailment_model
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
    the memory network, using the same fancy concatenation seen in the memory network solver.

    Note that this will only train the first knowledge selector.  We should probably re-use the
    layers, though, actually...  Pradeep: shouldn't we be doing that?  Using the same layers for
    the knowledge selector and the memory updater at each memory step?
    """
    # While it's not great, we need access to a few of the internals of the solver, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, solver, snli_file, **kwargs):
        super(SnliAttentionPretrainer, self).__init__(solver, snli_file, **kwargs)
        self.loss = 'binary_crossentropy'

    @overrides
    def _load_dataset(self):
        dataset = super(SnliAttentionPretrainer, self)._load_dataset()
        instances = [x.to_attention_instance() for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _get_model(self):
        sentence_shape = (self.solver.max_sentence_length,)
        text_input, embedded_text = self.solver._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.solver._get_embedded_sentence_input(sentence_shape,
                                                                                         "hypothesis")
        sentence_encoder = self.solver._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)

        merge_mode = lambda x: K.concatenate([K.expand_dims(x[0], dim=1), K.expand_dims(x[1], dim=1)],
                                             axis=1)
        merged_encoded_rep = merge([hypothesis_encoding, text_encoding],
                                   mode=merge_mode,
                                   output_shape=(2, self.solver.embedding_size),
                                   name='concat_hypothesis_with_text')
        knowledge_selector = self.solver._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(merged_encoded_rep)
        input_layers = [text_input, hypothesis_input]
        return Model(input=input_layers, output=attention_weights)
