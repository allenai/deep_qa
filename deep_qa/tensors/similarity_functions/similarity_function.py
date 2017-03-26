"""
Similarity functions take a pair of tensors with the same shape, and compute a similarity function
on the vectors in the last dimension.  For example, the tensors might both have shape
`(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
tensor of shape `(batch_size, sentence_length)`.

The similarity function could be as simple as a dot product, or it could be a more complex,
parameterized function.  The SimilarityFunction class exposes an API for a Layer that wants to
allow for multiple similarity functions, such as for initializing and returning weights.

If you want to compute a similarity between tensors of different sizes, you need to first tile them
in the appropriate dimensions to make them the same before you can use these functions.  The
Attention and MatrixAttention layers do this.
"""
from typing import List

from keras import activations, initializers

class SimilarityFunction:
    def __init__(self, name: str, initialization: str='glorot_uniform', activation: str='linear'):
        self.name = name
        self.init = initializers.get(initialization)
        self.activation = activations.get(activation)

    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        """
        Called in a `Layer.build()` method that uses this SimilarityFunction, here we both
        initialize whatever weights are necessary for this similarity function, and return them so
        they can be included in `Layer.trainable_weights`.


        Parameters
        ----------
        tensor_1_dim : int
            The last dimension (typically ``embedding_dim``) of the first input tensor.  We need
            this so we can initialize weights appropriately.
        tensor_2_dim : int
            The last dimension (typically ``embedding_dim``) of the second input tensor.  We need
            this so we can initialize weights appropriately.
        """
        raise NotImplementedError

    def compute_similarity(self, tensor_1, tensor_2):
        """
        Takes two tensors of the same shape, such as (batch_size, length_1, length_2,
        embedding_dim).  Computes a (possibly parameterized) similarity on the final dimension and
        returns a tensor with one less dimension, such as (batch_size, length_1, length_2).
        """
        raise NotImplementedError
