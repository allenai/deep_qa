from copy import deepcopy
from typing import Any, Dict

from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...common.params import pop_choice
from ...tensors.similarity_functions import similarity_functions


class MatrixAttention(MaskedLayer):
    '''
    This ``Layer`` takes two matrices as input and returns a matrix of attentions.

    We compute the similarity between each row in each matrix and return unnormalized similarity
    scores.  We don't worry about zeroing out any masked values, because we propagate a correct
    mask.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    This is largely similar to using ``TimeDistributed(Attention)``, except the result is
    unnormalized, and we return a mask, so you can do a masked normalization with the result.  You
    should use this instead of ``TimeDistributed(Attention)`` if you want to compute multiple
    normalizations of the attention matrix.

    Input:
        - matrix_1: ``(batch_size, num_rows_1, embedding_dim)``, with mask
          ``(batch_size, num_rows_1)``
        - matrix_2: ``(batch_size, num_rows_2, embedding_dim)``, with mask
          ``(batch_size, num_rows_2)``

    Output:
        - ``(batch_size, num_rows_1, num_rows_2)``, with mask of same shape

    Parameters
    ----------
    similarity_function_params: Dict[str, Any], default={}
        These parameters get passed to a similarity function (see
        :mod:`deep_qa.tensors.similarity_functions` for more info on what's acceptable).  The
        default similarity function with no parameters is a simple dot product.
    '''
    def __init__(self, similarity_function: Dict[str, Any]=None, **kwargs):
        super(MatrixAttention, self).__init__(**kwargs)
        self.similarity_function_params = deepcopy(similarity_function)
        if similarity_function is None:
            similarity_function = {}
        sim_function_choice = pop_choice(similarity_function, 'type',
                                         list(similarity_functions.keys()),
                                         default_to_first_choice=True)
        similarity_function['name'] = self.name + '_similarity_function'
        self.similarity_function = similarity_functions[sim_function_choice](**similarity_function)

    @overrides
    def build(self, input_shape):
        tensor_1_dim = input_shape[0][-1]
        tensor_2_dim = input_shape[1][-1]
        self.trainable_weights = self.similarity_function.initialize_weights(tensor_1_dim, tensor_2_dim)
        super(MatrixAttention, self).build(input_shape)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        mask_1, mask_2 = mask
        if mask_1 is None and mask_2 is None:
            return None
        if mask_1 is None:
            mask_1 = K.ones_like(K.sum(inputs[0], axis=-1))
        if mask_2 is None:
            mask_2 = K.ones_like(K.sum(inputs[1], axis=-1))
        # Theano can't do batch_dot on ints, so we need to cast to float and then back.
        mask_1 = K.cast(K.expand_dims(mask_1, axis=2), 'float32')
        mask_2 = K.cast(K.expand_dims(mask_2, axis=1), 'float32')
        return K.cast(K.batch_dot(mask_1, mask_2), 'uint8')

    @overrides
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    @overrides
    def call(self, inputs, mask=None):
        """
        NOTE: This does not work if ``num_rows_1`` or ``num_rows_2`` is ``None``!  I tried to get
        it to work, but ``K.dot()`` breaks.
        """
        matrix_1, matrix_2 = inputs
        matrix_1_shape = K.int_shape(matrix_1)
        matrix_2_shape = K.int_shape(matrix_2)
        num_rows_1 = matrix_1_shape[1]
        num_rows_2 = matrix_2_shape[1]
        tiled_matrix_1 = K.repeat_elements(K.expand_dims(matrix_1, axis=2), num_rows_2, axis=2)
        tiled_matrix_2 = K.repeat_elements(K.expand_dims(matrix_2, axis=1), num_rows_1, axis=1)

        # We need to be able to access K.int_shape() in compute_similarity() below, but in theano,
        # calling a backend function makes it so you can't use K.int_shape() anymore.  Setting
        # tensor._keras_shape here fixes that.
        # pylint: disable=protected-access
        tiled_matrix_1._keras_shape = matrix_1_shape[:2] + (num_rows_2,) + matrix_1_shape[2:]
        tiled_matrix_2._keras_shape = matrix_2_shape[:1] + (num_rows_1,) + matrix_2_shape[1:]
        return self.similarity_function.compute_similarity(tiled_matrix_1, tiled_matrix_2)

    @overrides
    def get_config(self):
        base_config = super(MatrixAttention, self).get_config()
        config = {'similarity_function': self.similarity_function_params}
        config.update(base_config)
        return config
