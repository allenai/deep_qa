from typing import List, Tuple

from keras import backend as K
from overrides import overrides

from .masked_layer import MaskedLayer
from ..common.checks import ConfigurationError


class ComplexConcat(MaskedLayer):
    """
    This ``Layer`` does ``K.concatenate()`` on a collection of tensors, but
    allows for more complex operations than ``Merge(mode='concat')``.
    Specifically, you can perform an arbitrary number of elementwise linear
    combinations of the vectors, and concatenate all of the results.  If you do
    not need to do this, you should use the regular ``Merge`` layer instead of
    this ``ComplexConcat``.

    Because the inputs all have the same shape, we assume that the masks are
    also the same, and just return the first mask.

    Input:
        - A list of tensors.  The tensors that you combine **must** have the
          same shape, so that we can do elementwise operations on them, and
          all tensors must have the same number of dimensions, and match on
          all dimensions except the concatenation axis.

    Output:
        - A tensor with some combination of the input tensors concatenated
          along a specific dimension.

    Parameters
    ----------
    axis : int
        The axis to use for ``K.concatenate``.

    combination: List of str
        A comma-separated list of combinations to perform on the input tensors.
        These are either tensor indices (1-indexed), or an arithmetic
        operation between two tensor indices (valid operations: ``*``, ``+``,
        ``-``, ``/``).  For example, these are all valid combination
        parameters: ``"1,2"``, ``"1,2*3"``, ``"1-2,2-1"``, ``"1,1*1"``,
        and ``"1,2,1*2"``.
    """
    def __init__(self, combination: str, axis: int=-1, **kwargs):
        self.axis = axis
        self.combination = combination
        self.combinations = self.combination.split(",")
        self.num_combinations = len(self.combinations)
        super(ComplexConcat, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return mask[0]

    @overrides
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            raise ConfigurationError("ComplexConcat input must be a list")
        output_shape = list(input_shape[0])
        output_shape[self.axis] = 0
        for combination in self.combinations:
            output_shape[self.axis] += self._get_combination_length(combination, input_shape)
        return tuple(output_shape)

    @overrides
    def call(self, x, mask=None):
        combined_tensor = self._get_combination(self.combinations[0], x)
        for combination in self.combinations[1:]:
            to_concatenate = self._get_combination(combination, x)
            combined_tensor = K.concatenate([combined_tensor, to_concatenate], axis=self.axis)
        return combined_tensor

    def _get_combination(self, combination: str, tensors: List['Tensor']):
        if combination.isdigit():
            return tensors[int(combination) - 1]  # indices in the combination string are 1-indexed
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_tensor = self._get_combination(combination[0], tensors)
            second_tensor = self._get_combination(combination[2], tensors)
            if K.int_shape(first_tensor) != K.int_shape(second_tensor):
                shapes_message = "Shapes were: {} and {}".format(K.int_shape(first_tensor),
                                                                 K.int_shape(second_tensor))
                raise ConfigurationError("Cannot combine two tensors with different shapes!  " +
                                         shapes_message)
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

    def _get_combination_length(self, combination: str, input_shapes: List[Tuple[int]]):
        if combination.isdigit():
            # indices in the combination string are 1-indexed
            return input_shapes[int(combination) - 1][self.axis]
        else:
            if len(combination) != 3:
                raise ConfigurationError("Invalid combination: " + combination)
            first_length = self._get_combination_length(combination[0], input_shapes)
            second_length = self._get_combination_length(combination[2], input_shapes)
            if first_length != second_length:
                raise ConfigurationError("Cannot combine two tensors with different shapes!")
            return first_length

    @overrides
    def get_config(self):
        config = {"combination": self.combination,
                  "axis": self.axis,
                 }
        base_config = super(ComplexConcat, self).get_config()
        config.update(base_config)
        return config
