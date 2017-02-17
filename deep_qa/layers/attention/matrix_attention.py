from keras import backend as K
from keras.layers import Layer

from ...common.params import get_choice_with_default
from ...tensors.similarity_functions import similarity_functions


class MatrixAttention(Layer):
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
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        # We need to wait until below to actually handle this, because self.name gets set in
        # super.__init__.
        similarity_function_params = kwargs.pop('similarity_function', {})
        super(MatrixAttention, self).__init__(**kwargs)
        sim_function_choice = get_choice_with_default(similarity_function_params,
                                                      'type',
                                                      list(similarity_functions.keys()))
        similarity_function_params['name'] = self.name + '_similarity_function'
        self.similarity_function = similarity_functions[sim_function_choice](**similarity_function_params)

    def build(self, input_shape):
        similarity_function_shape = self.get_output_shape_for(input_shape) + (input_shape[0][-1],)
        self.trainable_weights = self.similarity_function.initialize_weights(similarity_function_shape)
        super(MatrixAttention, self).build(input_shape)

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
        mask_1 = K.cast(K.expand_dims(mask_1, dim=2), 'float32')
        mask_2 = K.cast(K.expand_dims(mask_2, dim=1), 'float32')
        return K.cast(K.batch_dot(mask_1, mask_2), 'uint8')

    def get_output_shape_for(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[1][1])

    def call(self, inputs, mask=None):
        """
        NOTE: This does not work if ``num_rows_1`` or ``num_rows_2`` is ``None``!  I tried to get
        it to work, but ``K.dot()`` breaks.
        """
        matrix_1, matrix_2 = inputs
        num_rows_1 = K.int_shape(matrix_1)[1]
        num_rows_2 = K.int_shape(matrix_2)[1]
        tiled_matrix_1 = K.repeat_elements(K.expand_dims(matrix_1, dim=2), num_rows_2, axis=2)
        tiled_matrix_2 = K.repeat_elements(K.expand_dims(matrix_2, dim=1), num_rows_1, axis=1)
        return self.similarity_function.compute_similarity(tiled_matrix_1, tiled_matrix_2)
