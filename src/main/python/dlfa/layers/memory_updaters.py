"""
Memory updaters take a current memory vector and an aggregated background knowledge vector and
combine them to produce an updated memory vector.

Both the input vectors should have the same dimensionality, and the output vector should match that
dimensionality.
"""
from keras.layers import Dense, Layer


class SumMemoryUpdater(Layer):
    """
    This MemoryUpdater adds the memory vector and the aggregated knowledge vector.

    We can't just do a merge() here because we want to be able to TimeDistribute this layer, so we
    need to do some fancy footwork with the input vector.
    """
    def __init__(self, encoding_dim, name="sum_memory_updater"):
        super(SumMemoryUpdater, self).__init__(name=name)
        self.encoding_dim = encoding_dim
        self.mode = 'sum'

    def call(self, x, mask=None):
        memory_vector = x[:, :self.encoding_dim]
        aggregated_knowledge_vector = x[:, self.encoding_dim:]
        return memory_vector + aggregated_knowledge_vector

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], int(input_shape[1] / 2))


class DenseConcatMemoryUpdater(Dense):
    """
    This MemoryUpdater concatenates the memory vector and the aggregated knowledge vector, then
    passes them through a Dense layer.

    Because the input to the memory updater is already concatenated, we don't have to do anything
    here, we just subclass Dense.
    """
    def __init__(self, encoding_dim, name="dense_concat_memory_updater"):
        super(DenseConcatMemoryUpdater, self).__init__(encoding_dim, name=name)


updaters = {  # pylint: disable=invalid-name
        'dense_concat': DenseConcatMemoryUpdater,
        'sum': SumMemoryUpdater,
        }
