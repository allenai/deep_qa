from typing import List

from keras import backend as K
from overrides import overrides

from .similarity_function import SimilarityFunction


class Bilinear(SimilarityFunction):
    """
    This similarity function performs a bilinear transformation of the two input vectors.  This
    function has a matrix of weights W and a bias b, and the similarity between two vectors x and y
    is computed as `x^T W y + b`.
    """
    def __init__(self, **kwargs):
        super(Bilinear, self).__init__(**kwargs)
        self.weight_matrix = None
        self.bias = None

    @overrides
    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        self.weight_matrix = K.variable(self.init((tensor_1_dim, tensor_2_dim)),
                                        name=self.name + "_weights")
        self.bias = K.variable(self.init((1,)), name=self.name + "_bias")
        return [self.weight_matrix, self.bias]

    @overrides
    def compute_similarity(self, tensor_1, tensor_2):
        dot_product = K.sum(K.dot(tensor_1, self.weight_matrix) * tensor_2, axis=-1)
        return self.activation(dot_product + self.bias)
