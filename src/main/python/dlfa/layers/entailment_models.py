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

from keras.layers import Dense


class DenseMemoryOnlyEntailmentModel(object):
    @staticmethod
    def classify(sentence_encoding, current_memory, attended_background_knowledge):
        # pylint: disable=unused-argument
        return Dense(output_dim=2, activation='softmax', name='entailment_softmax')(current_memory)


entailment_models = {  # pylint: disable=invalid-name
        'dense_memory_only': DenseMemoryOnlyEntailmentModel
        }
