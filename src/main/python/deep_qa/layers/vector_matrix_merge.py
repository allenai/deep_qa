from keras import backend as K
from keras.layers import Layer


class VectorMatrixMerge(Layer):
    """
    This Layer takes a tensor with K modes and a collection of other tensors with K - 1 modes, and
    concatenates the lower-order tensors at the beginning of the higher-order tensor along a
    given mode.  We call this a vector-matrix merge to evoke the notion of appending vectors onto a
    matrix, but this will also work with higher-order tensors.

    For example, if you have a memory tensor of shape (batch_size, knowledge_length, encoding_dim),
    containing `knowledge_length` encoded sentences, you could use this layer to concatenate `N`
    individual encoded sentences with it, resulting in a tensor of shape
    (batch_size, N + knowledge_length, encoding_dim).

    This layer supports masking - we will pass through whatever mask you have on the matrix, and
    concatenate ones to it, similar to how to we concatenate the inputs.  We need to know what axis
    to do that concatenation on, though - we'll default to the input concatenation axis, but you
    can specify a different one if you need to.  We just ignore masks on the vectors, because doing
    the right thing with masked vectors here is complicated.  If you want to handle that later,
    submit a PR.
    """
    def __init__(self,
                 concat_axis: int,
                 mask_concat_axis: int=None,
                 propagate_mask: bool=True,
                 **kwargs):
        self.supports_masking = True
        self.concat_axis = concat_axis
        self.mask_concat_axis = mask_concat_axis if mask_concat_axis is not None else concat_axis
        self.propagate_mask = propagate_mask
        super(VectorMatrixMerge, self).__init__(**kwargs)

    def call(self, inputs, mask=None):
        # We need to reverse these here, so that the order is preserved when we roll out the
        # concatenations.
        vectors = reversed(inputs[:-1])
        matrix = inputs[-1]
        result = matrix
        for vector in vectors:
            expanded_vector = K.expand_dims(vector, dim=self.concat_axis)
            result = K.concatenate([expanded_vector, result], axis=self.concat_axis)
        return result

    def get_output_shape_for(self, input_shapes):
        num_vectors = len(input_shapes) - 1
        matrix_shape = input_shapes[-1]
        new_shape = list(matrix_shape)
        new_shape[self.concat_axis] += num_vectors
        return tuple(new_shape)

    def compute_mask(self, inputs, input_masks=None):  # pylint: disable=unused-argument
        if input_masks is None or not self.propagate_mask:
            return None
        num_vectors = len(input_masks) - 1
        matrix_mask = input_masks[-1]
        # We're just trying to get a mask of the right shape, here, so we can call K.ones_like() on
        # it.  We'll ignore the actual values in this.
        vector_mask_template = K.sum(matrix_mask, axis=self.mask_concat_axis)
        vector_mask = K.expand_dims(K.ones_like(vector_mask_template), dim=self.mask_concat_axis)
        result_mask = matrix_mask
        for _ in range(num_vectors):
            result_mask = K.concatenate([vector_mask, result_mask], axis=self.mask_concat_axis)
        return result_mask

    def get_config(self):
        base_config = super(VectorMatrixMerge, self).get_config()
        config = {
                'concat_axis': self.concat_axis,
                'mask_concat_axis': self.mask_concat_axis,
                'propagate_mask': self.propagate_mask,
                }
        config.update(base_config)
        return config
