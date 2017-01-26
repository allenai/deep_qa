from keras import backend as K
from keras.layers import Layer

from ...common.params import get_choice_with_default
from ...tensors.masked_operations import masked_softmax
from ...tensors.similarity_functions import similarity_functions


class Attention(Layer):
    '''
    This Layer takes two inputs: a vector and a matrix.  We compute the similarity between the
    vector and each row in the matrix, and then perform a softmax over rows using those computed
    similarities.  We handle masking properly for masked rows in the matrix, though we ignore any
    masking on the vector.

    By default similarity is computed with a dot product, but you can alternatively use a
    parameterized similarity function if you wish.

    Input shapes:
        vector: (batch_size, embedding_dim), mask is ignored if provided
        matrix: (batch_size, num_rows, embedding_dim), with mask (batch_size, num_rows)
    Output shape: (batch_size, num_rows), no mask (masked input rows have value 0 in the output)
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        # We need to wait until below to actually handle this, because self.name gets set in
        # super.__init__.
        similarity_function_params = kwargs.pop('similarity_function', {})
        super(Attention, self).__init__(**kwargs)
        sim_function_choice = get_choice_with_default(similarity_function_params,
                                                      'type',
                                                      list(similarity_functions.keys()))
        similarity_function_params['name'] = self.name + '_similarity_function'
        self.similarity_function = similarity_functions[sim_function_choice](**similarity_function_params)

    def build(self, input_shape):
        self.trainable_weights = self.similarity_function.initialize_weights(input_shape)
        super(Attention, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        # We do not need a mask beyond this layer.
        return None

    def get_output_shape_for(self, input_shapes):
        return (input_shapes[1][0], input_shapes[1][1])

    def call(self, inputs, mask=None):
        vector, matrix = inputs
        matrix_mask = mask[1]
        num_rows = K.int_shape(matrix)[1]
        print("Matrix shape:", K.int_shape(matrix))
        tiled_vector = K.repeat_elements(K.expand_dims(vector, dim=1), num_rows, axis=1)
        similarities = self.similarity_function.compute_similarity(tiled_vector, matrix)
        return masked_softmax(similarities, matrix_mask)
