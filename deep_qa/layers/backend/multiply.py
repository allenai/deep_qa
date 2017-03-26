from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class Multiply(MaskedLayer):
    """
    This ``Layer`` performs elementwise multiplication between two tensors, supporting masking.  We
    literally just call ``tensor_1 * tensor_2``; the only reason this is a ``Layer`` is so that we
    can support masking (and because it's slightly nicer to read in a model definition than a
    lambda layer).

    We also try to be a little bit smart if you're wanting to broadcast the multiplication, by
    having the tensors differ in the number of dimensions by one.

    Input:
        - tensor_1: a tensor of arbitrary shape, with an optional mask of the same shape
        - tensor_2: a tensor with the same shape as ``tensor_1`` (or one less or one more
          dimension), with an optional mask of the same shape

    Output:
        - ``tensor_1 * tensor_2``.
    """
    def __init__(self, **kwargs):
        super(Multiply, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        tensor_1, tensor_2 = inputs
        tensor_1_mask, tensor_2_mask = mask
        if tensor_1_mask is None:
            tensor_1_mask = K.ones_like(tensor_1)
        if tensor_2_mask is None:
            tensor_2_mask = K.ones_like(tensor_2)
        tensor_1_mask, tensor_2_mask = self.expand_dims_if_necessary(tensor_1_mask, tensor_2_mask)
        return K.cast(tensor_1_mask, 'uint8') * K.cast(tensor_2_mask, 'uint8')

    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @overrides
    def call(self, inputs, mask=None):
        tensor_1, tensor_2 = inputs
        tensor_1, tensor_2 = self.expand_dims_if_necessary(tensor_1, tensor_2)
        return tensor_1 * tensor_2

    @staticmethod
    def expand_dims_if_necessary(tensor_1, tensor_2):
        tensor_1_ndim = K.ndim(tensor_1)
        tensor_2_ndim = K.ndim(tensor_2)
        if tensor_1_ndim == tensor_2_ndim:
            return tensor_1, tensor_2
        elif tensor_1_ndim == tensor_2_ndim - 1:
            return K.expand_dims(tensor_1), tensor_2
        elif tensor_2_ndim == tensor_1_ndim - 1:
            return tensor_1, K.expand_dims(tensor_2)
        else:
            raise RuntimeError("Can't multiply two tensors with ndims "
                               "{} and {}".format(tensor_1_ndim, tensor_2_ndim))
