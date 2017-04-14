from keras import backend as K
from overrides import overrides

from .masked_layer import MaskedLayer


class VectorMatrixSplit(MaskedLayer):
    """
    This Layer takes a tensor with K modes and splits it into a tensor with K - 1 modes and a
    tensor with K modes, but one less row in one of the dimensions.  We call this a vector-matrix
    split to evoke the notion of taking a row- (or column-) vector off of a matrix and returning
    both the vector and the remaining matrix, but this will also work with higher-order tensors.

    For example, if you have a sentence that has a combined (word + characters) representation of
    the tokens in the sentence, you'd have a tensor of shape
    (batch_size, sentence_length, word_length + 1).  You could split that using this Layer into a
    tensor of shape (batch_size, sentence_length) for the word tokens in the sentence, and a tensor
    of shape (batch_size, sentence_length, word_length) for the character for each word token.

    This layer supports masking - we will split the mask the same way that we split the inputs.

    This Layer is essentially the opposite of a VectorMatrixMerge.
    """
    def __init__(self,
                 split_axis: int,
                 mask_split_axis: int=None,
                 propagate_mask: bool=True,
                 **kwargs):
        self.split_axis = split_axis
        self.mask_split_axis = mask_split_axis if mask_split_axis is not None else split_axis
        self.propagate_mask = propagate_mask
        super(VectorMatrixSplit, self).__init__(**kwargs)

    @overrides
    def call(self, inputs, mask=None):
        return self._split_tensor(inputs, self.split_axis)

    @overrides
    def compute_output_shape(self, input_shape):
        vector_shape = list(input_shape)
        del vector_shape[self.split_axis]
        matrix_shape = list(input_shape)
        if matrix_shape[self.split_axis] is not None:
            matrix_shape[self.split_axis] -= 1
        return [tuple(vector_shape), tuple(matrix_shape)]

    @overrides
    def compute_mask(self, inputs, input_mask=None):  # pylint: disable=unused-argument
        if input_mask is None or not self.propagate_mask:
            return [None, None]
        return self._split_tensor(input_mask, self.mask_split_axis)

    @staticmethod
    def _split_tensor(tensor, split_axis: int):
        modes = K.ndim(tensor)
        if split_axis < 0:
            split_axis = modes + split_axis
        vector_slice = []
        matrix_slice = []
        for mode in range(modes):
            if mode == split_axis:
                vector_slice.append(0)
                matrix_slice.append(slice(1, None, None))
            else:
                vector_slice.append(slice(None, None, None))
                matrix_slice.append(slice(None, None, None))
        return [tensor[vector_slice], tensor[matrix_slice]]

    @overrides
    def get_config(self):
        base_config = super(VectorMatrixSplit, self).get_config()
        config = {
                'split_axis': self.split_axis,
                'mask_split_axis': self.mask_split_axis,
                'propagate_mask': self.propagate_mask,
                }
        config.update(base_config)
        return config
