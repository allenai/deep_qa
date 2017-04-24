from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class CollapseToBatch(MaskedLayer):
    """
    Reshapes a higher order tensor, taking the first ``num_to_collapse`` dimensions after the batch
    dimension and folding them into the batch dimension.  For example, a tensor of shape (2, 4, 5,
    3), collapsed with ``num_to_collapse = 2``, would become a tensor of shape (40, 3).  We perform
    identical computation on the input mask, if there is one.

    This is essentially what Keras' ``TimeDistributed`` layer does (and then undoes) to apply a
    layer to a higher-order tensor, and that's the intended use for this layer.  However,
    ``TimeDistributed`` cannot handle distributing across dimensions with unknown lengths at graph
    compilation time.  This layer works even in that case.  So, if your actual tensor shape at
    graph compilation time looks like (None, None, None, 3), or (None, 4, None, 3), you can still
    use this layer (and :class:`~deep_qa.layers.backend.expand_from_batch.ExpandFromBatch`) to get
    the same result as ``TimeDistributed``.  If your shapes are fully known at graph compilation
    time, just use ``TimeDistributed``, as it's a nicer API for the same functionality.

    Inputs:
        - tensor with ``ndim >= 3``

    Output:
        - tensor with ``ndim = input_ndim - num_to_collapse``, with the removed dimensions folded
          into the first (batch-size) dimension

    Parameters
    ----------
    num_to_collapse: int
        The number of dimensions to fold into the batch size.
    """
    def __init__(self, num_to_collapse: int, **kwargs):
        self.num_to_collapse = num_to_collapse
        super(CollapseToBatch, self).__init__(**kwargs)

    @overrides
    def call(self, inputs, mask=None):
        return self.__collapse_tensor(inputs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return self.__collapse_tensor(mask)

    @overrides
    def compute_output_shape(self, input_shape):
        return (None,) + input_shape[1 + self.num_to_collapse:]

    @overrides
    def get_config(self):
        base_config = super(CollapseToBatch, self).get_config()
        config = {'num_to_collapse': self.num_to_collapse}
        config.update(base_config)
        return config

    def __collapse_tensor(self, tensor):
        # If we were to call K.int_shape(inputs), we would get something back that has None in it
        # (other than the batch dimension), because the shape is not fully known at graph
        # compilation time.  We can't do a reshape with more than one unknown dimension, which is
        # why we're doing this whole layer in the first place instead of just using
        # TimeDistributed.  tf.reshape will let us pass in a tensor that has the shape, instead of
        # just some ints.  So we can use tf.shape(tensor) to get the actual runtime shape of the
        # tensor _as a tensor_, which we then pass to tf.reshape().
        new_shape = K.concatenate([[-1], K.shape(tensor)[1 + self.num_to_collapse:]], 0)
        return K.reshape(tensor, new_shape)
