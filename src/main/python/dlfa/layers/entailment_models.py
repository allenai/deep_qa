"""
Entailment layers take three inputs, and combine them in some way to get to T/F.  They must end
with a softmax output over two dimensions.

Inputs:
    sentence_encoding
    current_memory
    attended_background_knowledge

We are trying to decide whether sentence_encoding is entailed by some background knowledge.
attended_background_knowledge and current_memory are two different vectors that encode the state of
reasoning over this background knowledge, which the entailment model can choose to use as desired.
"""

from keras import backend as K
from keras.layers import Dense, Layer, TimeDistributed, Lambda


class BasicEntailmentModel:
    """
    This entailment model assumes that we merge the three inputs in some way at the beginning into
    a single vector, then pass that vector through an MLP.  How we actually merge the three inputs
    is specified by one of the combiners below.
    """
    def __init__(self,
                 num_hidden_layers: int,
                 hidden_layer_width: int,
                 hidden_layer_activation: str):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

    def classify(self, combined_input, multiple_choice: bool=False):
        """
        Here we take the combined entailment input and decide whether it is true or false (or, in
        the case of multiple choice input, we do a softmax over the final entailment decisions for
        each choice).

        The combined input has shape (batch_size, combined_input_dim), or (batch_size, num_options,
        combined_input_dim) for multiple choice questions.  We do not know what combined_input_dim
        is, as it depends on the combiner.
        """
        # pylint: disable=redefined-variable-type
        hidden_input = combined_input
        for i in range(self.num_hidden_layers):
            layer = Dense(output_dim=self.hidden_layer_width,
                          activation=self.hidden_layer_activation,
                          name='entailment_hidden_layer_%d' % i)
            if multiple_choice:
                layer = TimeDistributed(layer)
            hidden_input = layer(hidden_input)
        if multiple_choice:
            # (batch_size, num_options, 1)
            score_layer = Dense(output_dim=1, activation='sigmoid', name='entailment_score')
            scores = TimeDistributed(score_layer)(hidden_input)
            softmax_layer = Lambda(lambda x: K.softmax(K.squeeze(scores, axis=2)),
                                   output_shape=lambda input_shape: (input_shape[0], input_shape[1]))
            softmax_output = softmax_layer(scores)
        else:
            softmax_layer = Dense(output_dim=2, activation='softmax', name='entailment_softmax')
            softmax_output = softmax_layer(hidden_input)
        return softmax_output


def split_combiner_inputs(x, encoding_dim: int):  # pylint: disable=invalid-name
    sentence_encoding = x[:, :encoding_dim]
    current_memory = x[:, encoding_dim:-encoding_dim]
    attended_knowledge = x[:, -encoding_dim:]
    return sentence_encoding, current_memory, attended_knowledge


class MemoryOnlyCombiner(Layer):
    """
    This "combiner" just selects the current memory and returns it.
    """
    def __init__(self, encoding_dim, name="entailment_combiner"):
        super(MemoryOnlyCombiner, self).__init__(name=name)
        self.encoding_dim = encoding_dim

    def call(self, x, mask=None):
        _, current_memory, _ = split_combiner_inputs(x, self.encoding_dim)
        return current_memory

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.encoding_dim)


class HeuristicMatchingCombiner(Layer):
    """
    This class is an implementation of the heuristic matching algorithm proposed in the following paper:
    "Natural Language Inference by Tree-Based Convolution and Heuristic Matching", Mou et al, ACL 2016.
    """
    def __init__(self, encoding_dim, name="entailment_combiner"):
        super(HeuristicMatchingCombiner, self).__init__(name=name)
        self.encoding_dim = encoding_dim

    def call(self, x, mask=None):
        sentence_encoding, current_memory, _ = split_combiner_inputs(x, self.encoding_dim)
        sentence_memory_product = sentence_encoding * current_memory
        sentence_memory_diff = sentence_encoding - current_memory
        return K.concatenate([sentence_encoding,
                              current_memory,
                              sentence_memory_product,
                              sentence_memory_diff])

    def get_output_shape_for(self, input_shape):
        # There are four components we've concatenated above: (1) the sentence encoding, (2) the
        # current memory, (3) the elementwise product of these two, and (4) their difference.  Each
        # of these has dimension `self.encoding_dim`.
        return (input_shape[0], self.encoding_dim * 4)


entailment_models = {  # pylint: disable=invalid-name
        'basic_mlp': BasicEntailmentModel,
        }

entailment_input_combiners = {  # pylint: disable=invalid-name
        'memory_only': MemoryOnlyCombiner,
        'heuristic_matching': HeuristicMatchingCombiner,
        }
