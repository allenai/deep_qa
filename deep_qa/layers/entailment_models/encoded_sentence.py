"""
Encoded sentence entailment layers take three inputs, and combine them in some way to get to T/F.
They must end with a softmax output over two dimensions.

Inputs:
    sentence_encoding
    current_memory
    attended_background_knowledge

We are trying to decide whether sentence_encoding is entailed by some background knowledge.
attended_background_knowledge and current_memory are two different vectors that encode the state of
reasoning over this background knowledge, which the entailment model can choose to use as desired.
"""

from keras import backend as K
from keras.layers import Dense, Layer, TimeDistributed

from ..attention.masked_softmax import MaskedSoftmax
from ..attention.attention import Attention


class TrueFalseEntailmentModel:
    """
    This entailment model assumes that we merge the three inputs in some way at the beginning into
    a single vector, then pass that vector through an MLP.  How we actually merge the three inputs
    is specified by one of the combiners below.
    """
    def __init__(self,
                 num_hidden_layers: int=1,
                 hidden_layer_width: int=50,
                 hidden_layer_activation: str="relu"):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

        self.hidden_layers = None
        self.softmax_layer = None
        self._init_layers()

    # TODO(matt): it would make sense to put the hidden layers and the score layer into a
    # superclass, as this method is the same for both the TrueFalseEntailmentModel and the
    # MultipleChoiceEntailmentModel.  The QuestionAnswerEntailmentModel throws a bit of a wrench
    # into this, though, so it needs some careful thought.
    def _init_layers(self):
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append(Dense(units=self.hidden_layer_width,
                                            activation=self.hidden_layer_activation,
                                            name='entailment_hidden_layer_%d' % i))
        self.score_layer = Dense(units=2, activation='softmax', name='entailment_softmax')

    def classify(self, combined_input):
        """
        Here we take the combined entailment input and decide whether it is true or false.  The
        combined input has shape (batch_size, combined_input_dim).  We do not know what
        combined_input_dim is, as it depends on the combiner.
        """
        hidden_input = combined_input
        for layer in self.hidden_layers:
            hidden_input = layer(hidden_input)
        softmax_output = self.score_layer(hidden_input)
        return softmax_output


class QuestionAnswerEntailmentModel:
    """
    In addition to the three combined inputs mentioned at the top of this file, this entailment
    model also takes a list of answer encodings.  Instead of going to a final softmax for the
    combined input, as the TrueFalseEntailmentModel does, this model projects the combined input
    into the same dimension as the answer encoding, does a dot product between the combined input
    encoding and the answer encoding, and does a final softmax over those similar scores.
    """
    def __init__(self,
                 answer_dim: int,
                 num_hidden_layers: int=1,
                 hidden_layer_width: int=50,
                 hidden_layer_activation: str="relu"):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation
        self.answer_dim = answer_dim

        self.hidden_layers = None
        self.softmax_layer = None
        self.projection_layer = None
        self.tile_layer = None
        self.similarity_layer = None
        self._init_layers()

    def _init_layers(self):
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append(Dense(units=self.hidden_layer_width,
                                            activation=self.hidden_layer_activation,
                                            name='entailment_hidden_layer_%d' % i))
        self.projection_layer = Dense(units=self.answer_dim,
                                      activation='linear',
                                      name='entailment_projection')

    def classify(self, combined_input, encoded_answers):
        """
        Here we take the combined entailment input, do a little more processing on it, then decide
        which answer it is closest to.  Specifically, we pass the combined input through a few
        hidden layers, then use a linear projection to get it into the same dimension as the answer
        encoding.  Then we do a dot product between the combined input and each of the answer
        options, and pass those values through a softmax.

        The combined input has shape (batch_size, combined_input_dim).  encoded_answers has shape
        (batch_size, num_options, answer_dim).
        """
        hidden_input = combined_input
        for layer in self.hidden_layers:
            hidden_input = layer(hidden_input)
        projected_input = self.projection_layer(hidden_input)

        # Note that this layer has no parameters, so it doesn't need to be put into self._init_layers().
        softmax_output = Attention(name='answer_similarity_softmax')([projected_input, encoded_answers])
        return softmax_output


class MultipleChoiceEntailmentModel:
    """
    This entailment model assumes that we merge the three inputs in some way at the beginning into
    a single vector, then pass that vector through an MLP, once for each of the multiple choices,
    then have a final softmax over answer options.
    """
    def __init__(self,
                 num_hidden_layers: int=1,
                 hidden_layer_width: int=50,
                 hidden_layer_activation: str="relu"):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

        self.hidden_layers = None
        self.score_layer = None
        self._init_layers()

    def _init_layers(self):
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append(Dense(units=self.hidden_layer_width,
                                            activation=self.hidden_layer_activation,
                                            name='entailment_hidden_layer_%d' % i))
        self.score_layer = Dense(units=1, activation='sigmoid')

    def classify(self, combined_input):
        """
        Here we take the combined entailment input for each option, decide whether it is true or
        false, then do a final softmax over the true/false scores for each option.

        The combined input has shape (batch_size, num_options, combined_input_dim).
        """
        # pylint: disable=redefined-variable-type
        hidden_input = combined_input
        for layer in self.hidden_layers:
            hidden_input = TimeDistributed(layer, name=layer.name)(hidden_input)

        # (batch_size, num_options, 1)
        scores = TimeDistributed(self.score_layer, name='entailment_scorer')(hidden_input)

        # This layer has no parameters, so it doesn't need to go into self._init_layers().
        softmax_layer = MaskedSoftmax(name='answer_option_softmax')
        softmax_output = softmax_layer(scores)
        return softmax_output


def split_combiner_inputs(inputs, encoding_dim: int):  # pylint: disable=invalid-name
    sentence_encoding = inputs[:, :encoding_dim]
    current_memory = inputs[:, encoding_dim:2*encoding_dim]
    attended_knowledge = inputs[:, 2*encoding_dim:]
    return sentence_encoding, current_memory, attended_knowledge


class MemoryOnlyCombiner(Layer):
    """
    This "combiner" just selects the current memory and returns it.
    """
    def __init__(self, encoding_dim, name="entailment_combiner", **kwargs):
        super(MemoryOnlyCombiner, self).__init__(name=name, **kwargs)
        self.encoding_dim = encoding_dim

    def call(self, inputs):
        _, current_memory, _ = split_combiner_inputs(inputs, self.encoding_dim)
        return current_memory

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.encoding_dim)

    def get_config(self):
        base_config = super(MemoryOnlyCombiner, self).get_config()
        config = {'encoding_dim': self.encoding_dim}
        config.update(base_config)
        return config


class HeuristicMatchingCombiner(Layer):
    """
    This class is an implementation of the heuristic matching algorithm proposed in the following paper:
    "Natural Language Inference by Tree-Based Convolution and Heuristic Matching", Mou et al, ACL 2016.
    """
    def __init__(self, encoding_dim, name="entailment_combiner", **kwargs):
        super(HeuristicMatchingCombiner, self).__init__(name=name, **kwargs)
        self.encoding_dim = encoding_dim

    def call(self, inputs):
        sentence_encoding, current_memory, _ = split_combiner_inputs(inputs, self.encoding_dim)
        sentence_memory_product = sentence_encoding * current_memory
        sentence_memory_diff = sentence_encoding - current_memory
        return K.concatenate([sentence_encoding,
                              current_memory,
                              sentence_memory_product,
                              sentence_memory_diff])

    def compute_output_shape(self, input_shape):
        # There are four components we've concatenated above: (1) the sentence encoding, (2) the
        # current memory, (3) the elementwise product of these two, and (4) their difference.  Each
        # of these has dimension `self.encoding_dim`.
        return (input_shape[0], self.encoding_dim * 4)

    def get_config(self):
        base_config = super(HeuristicMatchingCombiner, self).get_config()
        config = {'encoding_dim': self.encoding_dim}
        config.update(base_config)
        return config
