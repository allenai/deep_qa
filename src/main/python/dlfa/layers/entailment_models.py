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

from overrides import overrides

from keras.layers import Dense, merge


class BaseEntailmentModel:
    """
    This entailment model assumes that we merge the three inputs in some way at the beginning into
    a single vector, then pass that vector through an MLP.  How we actually merge the three inputs
    is not specified here, and must be implemented by a subclass.
    """
    def __init__(self, num_hidden_layers, hidden_layer_width, hidden_layer_activation):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation

    def classify(self, sentence_encoding, current_memory, attended_background_knowledge):
        mlp_input = self.combine_inputs(sentence_encoding, current_memory, attended_background_knowledge)
        hidden_input = mlp_input
        for i in range(self.num_hidden_layers):
            hidden_input = Dense(output_dim=self.hidden_layer_width,
                                 activation=self.hidden_layer_activation,
                                 name='entailment_hidden_layer_%d' % i)(hidden_input)
        return Dense(output_dim=2, activation='softmax', name='entailment_softmax')(hidden_input)

    def combine_inputs(self, sentence_encoding, current_memory, attended_background_knowledge):
        """
        Given the three input vectors, combine them in some way and return a single vector.
        """
        raise NotImplementedError


class DenseMemoryOnlyEntailmentModel(BaseEntailmentModel):
    """
    This entailment model is not really an entailment model.  It is what is used by standard memory
    networks and the attentive reader: just pass the current memory through an MLP (often just a
    linear MLP).
    """
    @overrides
    def combine_inputs(self, sentence_encoding, current_memory, attended_background_knowledge):
        # pylint: disable=unused-argument
        return current_memory


class HeuristicMatchingEntailmentModel:
    """
    This class is an implementation of the heuristic matching algorithm proposed in the following paper:
    "Natural Language Inference by Tree-Based Convolution and Heuristic Matching", Mou et al, ACL 2016.
    """
    @overrides
    def combine_inputs(self, sentence_encoding, current_memory, attended_background_knowledge):
        # pylint: disable=unused-argument,no-self-use
        sentence_memory_concat = merge([sentence_encoding, current_memory], mode="concat")
        sentence_memory_product = merge([sentence_encoding, current_memory], mode="mul")
        diff = lambda layers: layers[0] - layers[1]  # Keras does not have this merge mode
        diff_shape = lambda shapes: shapes[0]
        sentence_memory_diff = merge([sentence_encoding, current_memory], mode=diff,
                                     output_shape=diff_shape)
        return merge([sentence_memory_concat, sentence_memory_product, sentence_memory_diff],
                     mode="concat")


entailment_models = {  # pylint: disable=invalid-name
        'dense_memory_only': DenseMemoryOnlyEntailmentModel,
        'heuristic_matching': HeuristicMatchingEntailmentModel
        }
