from keras import backend as K
from keras.layers import Layer
from overrides import overrides


class Multiply(Layer):
    """
    This ``Layer`` performs elementwise multiplication between two tensors, supporting masking.  We
    literally just call ``tensor_1 * tensor_2``; the only reason this is a ``Layer`` is so that we
    can support masking (and because it's slightly nicer to read in a model definition than a
    lambda layer).

    Input:
        - tensor_1: a tensor of arbitrary shape, with an optional mask of the same shape
        - tensor_2: a tensor with the same shape as ``tensor_1``, with an optional mask of the same
          shape

    Output:
        - ``tensor_1 * tensor_2``.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
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
        return tensor_1_mask * tensor_2_mask

    @overrides
    def get_output_shape_for(self, input_shape):
        return input_shape[0]

    @overrides
    def call(self, x, mask=None):
        tensor_1, tensor_2 = x
        return tensor_1 * tensor_2
