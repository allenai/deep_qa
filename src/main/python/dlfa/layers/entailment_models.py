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

from keras.layers import Dense, merge


class DenseMemoryOnlyEntailmentModel(object):
    @staticmethod
    def classify(sentence_encoding, current_memory, attended_background_knowledge):
        # pylint: disable=unused-argument
        return Dense(output_dim=2, activation='softmax', name='entailment_softmax')(current_memory)

class HeuristicMatchingEntailmentModel(object):
    """
    This class is an implementation of the heuristic matching algorithm proposed in the following paper:
    "Natural Language Inference by Tree-Based Convolution and Heuristic Matching", Mou et al, ACL 2016
    """
    @staticmethod
    def classify(sentence_encoding, current_memory, attended_background_knowledge):
        # pylint: disable=unused-argument
        sentence_memory_concat = merge([sentence_encoding, current_memory], mode="concat")
        sentence_memory_product = merge([sentence_encoding, current_memory], mode="mul")
        diff = lambda layers: layers[0] - layers[1]  # Keras does not have this merge mode
        diff_shape = lambda shapes: shapes[0]
        sentence_memory_diff = merge([sentence_encoding, current_memory], mode=diff,
                                     output_shape=diff_shape)
        merged_input = merge([sentence_memory_concat, sentence_memory_product,
                              sentence_memory_diff], mode="concat")
        return Dense(output_dim=2, activation='softmax', name='entailment_softmax')(merged_input)


entailment_models = {  # pylint: disable=invalid-name
        'dense_memory_only': DenseMemoryOnlyEntailmentModel,
        'heuristic_matching': HeuristicMatchingEntailmentModel
        }
