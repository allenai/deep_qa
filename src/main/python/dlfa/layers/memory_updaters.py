"""
Memory updaters take a current memory vector and an aggregated background knowledge vector and
combine them to produce an updated memory vector.

Both the input vectors should have the same dimensionality, and the output vector should match that
dimensionality.
"""
from keras import backend as K
from keras.layers import Dense


class SumMemoryUpdater(object):
    @staticmethod
    def update(memory_vector, aggregated_knowledge_vector):
        return memory_vector + aggregated_knowledge_vector


class DenseConcatMemoryUpdater(object):
    @staticmethod
    def update(memory_vector, aggregated_knowledge_vector):
        # We're assuming the both memory_vector and aggregated_knowledge_vector have shape
        # (batch_size, output_dim).
        output_dim = memory_vector.shape[1]
        return Dense(output_dim)(K.concatenate([memory_vector, aggregated_knowledge_vector]))


updaters = {  # pylint: disable=invalid-name
        'dense_concat': None,
        'sum': None,
        }
