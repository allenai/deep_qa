from typing import List, Tuple

from keras import backend as K

from ...common.checks import ConfigurationError
from .similarity_function import SimilarityFunction


class Linear(SimilarityFunction):
    """
    This similarity function performs a dot product between a vector of weights and some
    combination of the two input vectors.  The combination done is configurable.

    If the two vectors are `x` and `y`, we allow the following kinds of combinations: `x`, `y`,
    `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations is performed elementwise.
    You can list as many combinations as you want, comma separated.  For example, you might give
    "x,y,x*y" as the `combination` parameter to this class.  The computed similarity function would
    then be `w^T [x; y; x*y] + b`, where `w` is a vector of weights, `b` is a bias parameter, and
    `[;]` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.
    """
    def __init__(self, combination: str='x,y', **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.combinations = combination.split(',')
        self.num_combinations = len(self.combinations)
        self.weight_vector = None
        self.bias = None

    def initialize_weights(self, input_shape: Tuple[int]) -> List['K.variable']:
        embedding_dim = input_shape[-1]
        self.weight_vector = self.init((self.num_combinations * embedding_dim, 1),
                                       name='{}_dense'.format(self.name))
        self.bias = self.init((1,), name='{}_bias'.format(self.name))
        return [self.weight_vector, self.bias]

    def compute_similarity(self, tensor_1, tensor_2):
        combined_tensors = self.combine_tensors(tensor_1, tensor_2)
        dot_product = K.squeeze(K.dot(combined_tensors, self.weight_vector), axis=-1)
        if K.backend() == 'theano':
            # For some reason theano is having a hard time broadcasting the elementwise addition,
            # so we need to do this repeat.
            bias = K.repeat_elements(self.bias, K.int_shape(tensor_1)[-2], 0)
        else:
            bias = self.bias
        return self.activation(dot_product + bias)

    def combine_tensors(self, tensor_1, tensor_2):
        combined_tensor = self.get_combination(self.combinations[0], tensor_1, tensor_2)
        for combination in self.combinations[1:]:
            to_concatenate = self.get_combination(combination, tensor_1, tensor_2)
            combined_tensor = K.concatenate([combined_tensor, to_concatenate], axis=-1)
        return combined_tensor

    def get_combination(self, combination, tensor_1, tensor_2):
        if combination == 'x':
            return tensor_1
        elif combination == 'y':
            return tensor_2
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor = self.get_combination(combination[0], tensor_1, tensor_2)
            second_tensor = self.get_combination(combination[2], tensor_1, tensor_2)
            operation = combination[1]
            if operation == '*':
                return first_tensor * second_tensor
            elif operation == '/':
                return first_tensor / second_tensor
            elif operation == '+':
                return first_tensor + second_tensor
            elif operation == '-':
                return first_tensor - second_tensor
            else:
                raise ConfigurationError("Invalid operation: " + operation)
