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


class TrueFalseEntailmentModel:
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

    def classify(self, combined_input):
        """
        Here we take the combined entailment input and decide whether it is true or false.  The
        combined input has shape (batch_size, combined_input_dim).  We do not know what
        combined_input_dim is, as it depends on the combiner.
        """
        hidden_input = combined_input
        for i in range(self.num_hidden_layers):
            layer = Dense(output_dim=self.hidden_layer_width,
                          activation=self.hidden_layer_activation,
                          name='entailment_hidden_layer_%d' % i)
            hidden_input = layer(hidden_input)
        softmax_layer = Dense(output_dim=2, activation='softmax', name='entailment_softmax')
        softmax_output = softmax_layer(hidden_input)
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
                 num_hidden_layers: int,
                 hidden_layer_width: int,
                 hidden_layer_activation: str):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

    def classify(self, combined_input, encoded_answers, answer_dim: int):
        """
        Here we take the combined entailment input and decide whether it is true or false.  The
        combined input has shape (batch_size, combined_input_dim).  We do not know what
        combined_input_dim is, as it depends on the combiner.
        """
        hidden_input = combined_input
        for i in range(self.num_hidden_layers):
            layer = Dense(output_dim=self.hidden_layer_width,
                          activation=self.hidden_layer_activation,
                          name='entailment_hidden_layer_%d' % i)
            hidden_input = layer(hidden_input)
        projection_layer = Dense(output_dim=answer_dim, activation='linear', name='entailment_projection')
        projected_input = projection_layer(hidden_input)

        def tile_projection(inputs):
            # We need to tile the projected_input so that we can easily do a dot product with the
            # encoded_answers.  This follows the logic in knowledge_selectors.tile_sentence_encoding.
            answers, projected = inputs
            # Shape: (num_options, batch_size, answer_dim)
            ones = K.permute_dimensions(K.ones_like(answers), [1, 0, 2])
            # Shape: (batch_size, num_options, answer_dim)
            return K.permute_dimensions(ones * projected, [1, 0, 2])

        tile_layer = Lambda(tile_projection,
                            output_shape=lambda input_shapes: input_shapes[0],
                            name='tile_entailment_input')
        tiled_projected_input = tile_layer([encoded_answers, projected_input])

        softmax_similarity = Lambda(lambda x: K.softmax(K.sum(x[0] * x[1], axis=2)),
                                    output_shape=lambda input_shapes: (input_shapes[0][0], input_shapes[0][1]),
                                    name='answer_similarity_softmax')
        # Shape: (batch_size, num_options)
        softmax_output = softmax_similarity([tiled_projected_input, encoded_answers])
        return softmax_output


class MultipleChoiceEntailmentModel:
    """
    This entailment model assumes that we merge the three inputs in some way at the beginning into
    a single vector, then pass that vector through an MLP, once for each of the multiple choices,
    then have a final softmax over answer options.
    """
    def __init__(self,
                 num_hidden_layers: int,
                 hidden_layer_width: int,
                 hidden_layer_activation: str):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

    def classify(self, combined_input):
        """
        Here we take the combined entailment input for each option, decide whether it is true or
        false, then do a final softmax over the true/false scores for each option.

        The combined input has shape (batch_size, num_options, combined_input_dim).
        """
        # pylint: disable=redefined-variable-type
        hidden_input = combined_input
        for i in range(self.num_hidden_layers):
            layer = TimeDistributed(Dense(output_dim=self.hidden_layer_width,
                                          activation=self.hidden_layer_activation),
                                    name='entailment_hidden_layer_%d' % i)
            hidden_input = layer(hidden_input)
        # (batch_size, num_options, 1)
        score_layer = TimeDistributed(Dense(output_dim=1, activation='sigmoid'), name='entailment_score')
        scores = score_layer(hidden_input)
        softmax_layer = Lambda(lambda x: K.softmax(K.squeeze(x, axis=2)),
                               output_shape=lambda input_shape: (input_shape[0], input_shape[1]),
                               name='answer_option_softmax')
        softmax_output = softmax_layer(scores)
        return softmax_output


def split_combiner_inputs(x, encoding_dim: int):  # pylint: disable=invalid-name
    sentence_encoding = x[:, :encoding_dim]
    current_memory = x[:, encoding_dim:2*encoding_dim]
    attended_knowledge = x[:, 2*encoding_dim:]
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
        'true_false_mlp': TrueFalseEntailmentModel,
        'multiple_choice_mlp': MultipleChoiceEntailmentModel,
        'question_answer_mlp': QuestionAnswerEntailmentModel,
        }

entailment_input_combiners = {  # pylint: disable=invalid-name
        'memory_only': MemoryOnlyCombiner,
        'heuristic_matching': HeuristicMatchingCombiner,
        }
