from keras import backend as K
from keras.layers import Layer

from ...common.tensors import masked_softmax

class SimilaritySoftmax(Layer):
    '''
    This Layer takes two inputs: a vector and a matrix.  We compute the similarity between the
    vector and each row in the matrix, and then perform a softmax over rows using those computed
    similarities.  This is how attentions are typically computed, so if you're computing an
    attention, you should consider using this layer.  We handle masking properly for masked rows in
    the matrix, though we ignore any masking on the vector.

    Input shapes:
        vector: (batch_size, embedding_dim), mask is ignored if provided
        matrix: (batch_size, num_rows, embedding_dim), with mask (batch_size, num_rows)
    Output shape: (batch_size, num_rows), no mask (masked input rows have value 0 in the output)
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(SimilaritySoftmax, self).__init__(**kwargs)

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1])

    def call(self, inputs, mask=None):
        vector, matrix = inputs
        matrix_mask = mask[1]
        vector = K.expand_dims(vector, dim=-1)
        similarities = K.batch_dot(vector, matrix, axes=(1, 2))
        similarities = K.squeeze(similarities, axis=1)
        softmax_output = masked_softmax(similarities, matrix_mask)
        return softmax_output
