from keras import backend as K
from keras.layers import Layer
from overrides import overrides

from ...tensors.backend import switch


class WeightedSum(Layer):
    '''
    This Layer takes a matrix of vectors and a vector of row weights, and returns a weighted sum
    of the vectors.  You might use this to get some aggregate sentence representation after
    computing an attention over the sentence, for example.

    Input shapes:
        matrix: (batch_size, num_rows, embedding_dim), with mask (batch_size, num_rows)
        vector: (batch_size, num_rows), mask is ignored
    Output shape: (batch_size, embedding_dim)

    A usage note: you probably should have used a mask when you computed your attention weights, so
    any row that's masked in the matrix _should_ already be 0 in the attention vector.  But just in
    case you didn't, we'll handle a mask on the matrix here too.

    While the above spec shows inputs with 3 and 2 modes, we also allow inputs of any order; we
    always sum over the second-to-last dimension of the "matrix", weighted by the last dimension of
    the "vector".  Higher-order tensors get complicated for matching things, though, so there is a
    hard constraint: all dimensions in the "matrix" before the final embedding must be matched in
    the "vector".

    For example, say I have a "matrix" with dimensions (batch_size, num_queries, num_words,
    embedding_dim), representing some kind of embedding or encoding of several multi-word queries.
    My attention "vector" must then have at least those dimensions, and could have more.  So I
    could have an attention over words per query, with shape (batch_size, num_queries, num_words),
    or I could have an attention over query words for every document in some list, with shape
    (batch_size, num_documents, num_queries, num_words).  Both of these cases are fine.  In the
    first case, the returned tensor will have shape (batch_size, num_queries, embedding_dim), and
    in the second case, it will have shape (batch_size, num_documents, num_queries, embedding_dim).
    But you _can't_ have an attention "vector" that does not include all of the queries, so shape
    (batch_size, num_words) is not allowed - you haven't specified how to handle that dimension in
    the "matrix", so we can't do anything with this input.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(WeightedSum, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We don't need to worry about a mask after we've summed over the rows of the matrix.
        # You might actually still need a mask if you used a higher-order tensor, but probably the
        # right place to handle that is with careful use of TimeDistributed.  Or submit a PR.
        return None

    @overrides
    def get_output_shape_for(self, input_shapes):
        matrix_shape, attention_shape = input_shapes
        return attention_shape[:-1] + matrix_shape[-1:]

    @overrides
    def call(self, inputs, mask=None):
        matrix, attention_vector = inputs
        matrix = self._expand_matrix_if_necessary(matrix, attention_vector)
        matrix_mask = mask[0]
        if matrix_mask is not None:
            matrix_mask = K.expand_dims(matrix_mask, dim=-1)
            matrix_mask = self._expand_matrix_if_necessary(matrix_mask, attention_vector)
            matrix = switch(matrix_mask, matrix, K.zeros_like(matrix))
        return K.sum(K.expand_dims(attention_vector, dim=-1) * matrix, -2)

    @staticmethod
    def _expand_matrix_if_necessary(matrix, attention_vector):
        try:
            matrix_shape = K.int_shape(matrix)[:-1]  # taking off the embedding dimension here.
        except Exception:  # pylint: disable=broad-except
            if K.backend() == 'theano':
                raise RuntimeError("Theano backend doesn't support K.int_shape very well - use "
                                   "tensorflow instead for this model.")
            else:
                raise
        attention_shape = K.int_shape(attention_vector)
        if matrix_shape != attention_shape:
            # We'll take care of the batch size first.  After this, the matrix_shape should match
            # the end of the attention_shape exactly.
            assert matrix_shape[0] == attention_shape[0], "somehow batch sizes don't match"
            matrix_shape = matrix_shape[1:]
            attention_shape = attention_shape[1:]
            assert attention_shape[-len(matrix_shape):] == matrix_shape, ("matrix_shape must be "
                                                                          "subset of attention_shape")
            for i in range(len(attention_shape) - len(matrix_shape)):
                matrix = K.expand_dims(matrix, dim=i+1)  # +1 to account for batch_size
                matrix = K.repeat_elements(matrix, attention_shape[i], axis=i+1)
        return matrix
