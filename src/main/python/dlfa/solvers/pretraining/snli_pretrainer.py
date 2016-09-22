from overrides import overrides

from keras import backend as K
from keras.layers import merge
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

    @overrides
    def _get_model(self):
        sentence_shape = (self.max_sentence_length,)
        text_input, embedded_text = self.solver._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.solver._get_embedded_sentence_input(sentence_shape,
                                                                                         "hypothesis")
        sentence_encoder = self.solver._get_sentence_encoder()
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)
        empty_background_encoding = K.zeros_like(text_encoding)
        entailment_input = merge([hypothesis_encoding, text_encoding, empty_background_encoding],
                                 mode='concat',
                                 concat_axis=1,
                                 name='concat_entailment_inputs')
        combined_input = self.solver._get_entailment_combiner()(entailment_input)
        extra_entailment_inputs, entailment_output = self.solver._get_entailment_output(combined_input)
        input_layers = [text_input, hypothesis_input]
        input_layers.extend(extra_entailment_inputs)
        return Model(input=input_layers, output=entailment_output)


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
        # TODO(matt): this isn't the right loss; just getting something to pass tests while I don't
        # have an internet connection to figure out the right loss.
        self.loss = 'mse'

    @overrides
    def _load_dataset(self):
        dataset = super(SnliAttentionPretrainer, self)._load_dataset()
        instances = [x.to_attention_instance() for x in dataset.instances]
        return TextDataset(instances)

    @overrides
    def _get_model(self):
        sentence_shape = (self.max_sentence_length,)
        text_input, embedded_text = self.solver._get_embedded_sentence_input(sentence_shape, "text")
        hypothesis_input, embedded_hypothesis = self.solver._get_embedded_sentence_input(sentence_shape,
                                                                                         "hypothesis")
        sentence_encoder = self.solver._get_sentence_encoder()
        text_encoding = sentence_encoder(embedded_text)
        hypothesis_encoding = sentence_encoder(embedded_hypothesis)

        merge_mode = lambda x: K.concatenate([K.expand_dims(x[0], dim=1), K.expand_dims(x[1], dim=1)],
                                             axis=1)
        merged_encoded_rep = merge([hypothesis_encoding, text_encoding],
                                   mode=merge_mode,
                                   output_shape=(2,) + sentence_shape,
                                   name='concat_hypothesis_with_text')
        knowledge_selector = self.solver._get_knowledge_selector(0)
        attention_weights = knowledge_selector(merged_encoded_rep)
        input_layers = [text_input, hypothesis_input]
        return Model(input=input_layers, output=attention_weights)
