"""
Memory updaters take a current memory vector and an aggregated background knowledge vector and
combine them to produce an updated memory vector.

Both the input vectors should have the same dimensionality, and the output vector should match that
dimensionality.
"""
from keras import backend as K
from keras.layers import Dense, merge


class SumMemoryUpdater:
    def __init__(self, output_dim, name="sum_memory_updater"):
        self.output_dim = output_dim
        self.name = name
    
    def update(self, memory_vector, aggregated_knowledge_vector):
        # Returning a Keras layer instead of symbolic sum so that we can name the output.
        return merge([memory_vector, aggregated_knowledge_vector], mode='sum',
                    output_shape=(self.output_dim,) name=self.name)


class DenseConcatMemoryUpdater:
    def __init__(self, output_dim, name="dense_concat_memory_updater"):
        self.output_dim = output_dim
        self.name = name
    
    def update(self, memory_vector, aggregated_knowledge_vector):
        # We're assuming the both memory_vector and aggregated_knowledge_vector have shape
        # (batch_size, output_dim).
        # We need the concatenation to be done by a layer to propagate the shape information.
        # Or else the following layer wouldn't know what shape to expect.
        concatenated_input = merge([memory_vector, aggregated_knowledge_vector], mode='concat')
        return Dense(self.output_dim, name=self.name)(concatenated_input)


updaters = {  # pylint: disable=invalid-name
        'dense_concat': DenseConcatMemoryUpdater,
        'sum': SumMemoryUpdater,
        }
