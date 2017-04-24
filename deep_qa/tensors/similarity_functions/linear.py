from typing import List

from keras import backend as K
from overrides import overrides

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

    @overrides
    def initialize_weights(self, tensor_1_dim: int, tensor_2_dim: int) -> List['K.variable']:
        combined_dim = self._get_combined_dim(tensor_1_dim, tensor_2_dim)
        self.weight_vector = K.variable(self.init((combined_dim, 1)), name=self.name + "_weights")
        self.bias = K.variable(self.init((1,)), name=self.name + "_bias")
        return [self.weight_vector, self.bias]

    @overrides
    def compute_similarity(self, tensor_1, tensor_2):
        combined_tensors = self._combine_tensors(tensor_1, tensor_2)
        dot_product = K.squeeze(K.dot(combined_tensors, self.weight_vector), axis=-1)
        return self.activation(dot_product + self.bias)

    def _combine_tensors(self, tensor_1, tensor_2):
        combined_tensor = self._get_combination(self.combinations[0], tensor_1, tensor_2)
        for combination in self.combinations[1:]:
            to_concatenate = self._get_combination(combination, tensor_1, tensor_2)
            combined_tensor = K.concatenate([combined_tensor, to_concatenate], axis=-1)
        return combined_tensor

    def _get_combination(self, combination: str, tensor_1, tensor_2):
        if combination == 'x':
            return tensor_1
        elif combination == 'y':
            return tensor_2
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor = self._get_combination(combination[0], tensor_1, tensor_2)
            second_tensor = self._get_combination(combination[2], tensor_1, tensor_2)
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

    def _get_combined_dim(self, tensor_1_dim: int, tensor_2_dim: int) -> int:
        combination_dims = [self._get_combination_dim(combination, tensor_1_dim, tensor_2_dim)
                            for combination in self.combinations]
        return sum(combination_dims)

    def _get_combination_dim(self, combination: str, tensor_1_dim: int, tensor_2_dim: int) -> int:
        if combination == 'x':
            return tensor_1_dim
        elif combination == 'y':
            return tensor_2_dim
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor_dim = self._get_combination_dim(combination[0], tensor_1_dim, tensor_2_dim)
            second_tensor_dim = self._get_combination_dim(combination[2], tensor_1_dim, tensor_2_dim)
            operation = combination[1]
            if first_tensor_dim != second_tensor_dim:
                raise ConfigurationError("Tensor dims must match for operation \"{}\"".format(operation))
            return first_tensor_dim
