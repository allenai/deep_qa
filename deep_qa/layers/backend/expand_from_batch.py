from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class ExpandFromBatch(MaskedLayer):
    """
    Reshapes a collapsed tensor, taking the batch size and separating it into ``num_to_expand``
    dimensions, following the shape of a second input tensor.  This is meant to be used in
    conjunction with :class:`~deep_qa.layers.backend.collapse_to_batch.CollapseToBatch`, to achieve
    the same effect as Keras' ``TimeDistributed`` layer, but for shapes that are not fully
    specified at graph compilation time.

    For example, say you had an original tensor of shape ``(None (2), 4, None (5), 3)``, then
    collapsed it with ``CollapseToBatch(2)(tensor)`` to get a tensor with shape ``(None (40), 3)``
    (here I'm using ``None (x)`` to denote a dimension with unknown length at graph compilation
    time, where ``x`` is the actual runtime length).  You can then call
    ``ExpandFromBatch(2)(collapsed, tensor)`` with the result to expand the first two dimensions
    out of the batch again (presumably after you've done some computation when it was collapsed).

    Inputs:
        - a tensor that has been collapsed with ``CollapseToBatch(num_to_expand)``.
        - the original tensor that was used as input to ``CollapseToBatch`` (or one with identical
          shape in the collapsed dimensions).  We will use this input only to get its shape.

    Output:
        - tensor with ``ndim = input_ndim + num_to_expand``, with the additional dimensions coming
          immediately after the first (batch-size) dimension.

    Parameters
    ----------
    num_to_expand: int
        The number of dimensions to expand from the batch size.
    """
    def __init__(self, num_to_expand: int, **kwargs):
        self.num_to_expand = num_to_expand
        super(ExpandFromBatch, self).__init__(**kwargs)

    @overrides
    def call(self, inputs, mask=None):
        return self.__reshape_tensors(inputs[0], inputs[1])

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        if mask[0] is None or mask[1] is None:
            return None
        return self.__reshape_tensors(mask[0], mask[1])

    @overrides
    def compute_output_shape(self, input_shape):
        collapsed_shape, original_shape = input_shape
        return (None,) + original_shape[1:1 + self.num_to_expand] + collapsed_shape[1:]

    @overrides
    def get_config(self):
        base_config = super(ExpandFromBatch, self).get_config()
        config = {'num_to_expand': self.num_to_expand}
        config.update(base_config)
        return config

    def __reshape_tensors(self, collapsed_tensor, original_tensor):
        collapsed_shape = K.shape(original_tensor)[1:1 + self.num_to_expand]
        remaining_shape = K.shape(collapsed_tensor)[1:]
        new_shape = K.concatenate([[-1], collapsed_shape, remaining_shape], 0)
        return K.reshape(collapsed_tensor, new_shape)
