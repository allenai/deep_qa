from keras import backend as K
from overrides import overrides

from .masked_layer import MaskedLayer


class VectorMatrixMerge(MaskedLayer):
    """
    This ``Layer`` takes a tensor with ``K`` modes and a collection of other tensors with ``K - 1``
    modes, and concatenates the lower-order tensors at the beginning of the higher-order tensor
    along a given mode.  We call this a vector-matrix merge to evoke the notion of appending
    vectors onto a matrix, but this will also work with higher-order tensors.

    For example, if you have a memory tensor of shape ``(batch_size, knowledge_length,
    encoding_dim)``, containing ``knowledge_length`` encoded sentences, you could use this layer to
    concatenate ``N`` individual encoded sentences with it, resulting in a tensor of shape
    ``(batch_size, N + knowledge_length, encoding_dim)``.

    This layer supports masking - we will pass through whatever mask you have on the matrix, and
    concatenate ones to it, similar to how to we concatenate the inputs.  We need to know what axis
    to do that concatenation on, though - we'll default to the input concatenation axis, but you
    can specify a different one if you need to.  We just ignore masks on the vectors, because doing
    the right thing with masked vectors here is complicated.  If you want to handle that later,
    submit a PR.

    This ``Layer`` is essentially the opposite of a
    :class:`~deep_qa.layers.vector_matrix_split.VectorMatrixSplit`.

    Parameters
    ----------
    concat_axis: int
        The axis to concatenate the vectors and matrix on.
    mask_concat_axis: int, optional (default=None)
        The axis to concatenate the masks on (defaults to ``concat_axis`` if ``None``)
    """
    def __init__(self,
                 concat_axis: int,
                 mask_concat_axis: int=None,
                 propagate_mask: bool=True,
                 **kwargs):
        self.concat_axis = concat_axis
        self.mask_concat_axis = mask_concat_axis
        self.propagate_mask = propagate_mask
        super(VectorMatrixMerge, self).__init__(**kwargs)

    @overrides
    def call(self, inputs, mask=None):
        # We need to reverse these here, so that the order is preserved when we roll out the
        # concatenations.
        vectors = inputs[:-1]
        matrix = inputs[-1]
        expanded_vectors = [K.expand_dims(vector, axis=self.concat_axis) for vector in vectors]
        return K.concatenate(expanded_vectors + [matrix], axis=self.concat_axis)

    @overrides
    def compute_output_shape(self, input_shapes):
        num_vectors = len(input_shapes) - 1
        matrix_shape = input_shapes[-1]
        new_shape = list(matrix_shape)
        new_shape[self.concat_axis] += num_vectors
        return tuple(new_shape)

    @overrides
    def compute_mask(self, inputs, mask=None):
        if mask is None or all(m is None for m in mask) or not self.propagate_mask:
            return None
        mask_concat_axis = self.mask_concat_axis
        if mask_concat_axis is None:
            mask_concat_axis = self.concat_axis
            if mask_concat_axis < 0:
                mask_concat_axis %= K.ndim(inputs[-1])
        num_vectors = len(mask) - 1
        matrix_mask = mask[-1]
        if mask_concat_axis >= K.ndim(matrix_mask):
            # This means we're concatenating along an axis in the tensor that is greater than the
            # number of dimensions in the mask.  E.g., we're adding a single pre-computed feature
            # to a word embedding (if it was multiple features, you'd already have evenly shaped
            # tensors, so you could just use a Concatenate layer).  In this case, we take all of
            # the masks, assume they have the same shape, and compute K.any() with them.
            masks = [matrix_mask] + [m for m in mask[:-1] if m is not None]
            shapes = set([K.int_shape(m) for m in masks])
            assert len(shapes) == 1, "Can't compute mask with uneven shapes: " + shapes
            expanded_masks = [K.expand_dims(m, axis=-1) for m in masks]
            concated_masks = K.concatenate(expanded_masks, axis=-1)
            return K.any(concated_masks, axis=-1)
        vector_masks = []
        for i in range(num_vectors):
            vector_mask = mask[i]
            if vector_mask is None:
                vector_mask_template = K.sum(K.cast(matrix_mask, 'uint8'), axis=mask_concat_axis)
                vector_mask = K.cast(K.ones_like(vector_mask_template), 'bool')
            vector_masks.append(K.expand_dims(vector_mask, axis=mask_concat_axis))
        return K.concatenate(vector_masks + [matrix_mask], axis=mask_concat_axis)

    @overrides
    def get_config(self):
        base_config = super(VectorMatrixMerge, self).get_config()
        config = {
                'concat_axis': self.concat_axis,
                'mask_concat_axis': self.mask_concat_axis,
                'propagate_mask': self.propagate_mask,
                }
        config.update(base_config)
        return config
