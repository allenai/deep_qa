from typing import List, Tuple

from keras import backend as K

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

    def initialize_weights(self, input_shape: Tuple[int]) -> List['K.variable']:
        embedding_dim = input_shape[-1]
        self.weight_matrix = self.init((embedding_dim, embedding_dim),
                                       name='{}_dense'.format(self.name))
        self.bias = self.init((1,), name='{}_bias'.format(self.name))
        return [self.weight_matrix, self.bias]

    def compute_similarity(self, tensor_1, tensor_2):
        dot_product = K.sum(K.dot(tensor_1, self.weight_matrix) * tensor_2, axis=-1)
        if K.backend() == 'theano':
            # For some reason theano is having a hard time broadcasting the elementwise addition,
            # so we need to do this repeat.
            bias = K.repeat_elements(self.bias, K.int_shape(tensor_1)[-2], 0)
        else:
            bias = self.bias
        return self.activation(dot_product + bias)
