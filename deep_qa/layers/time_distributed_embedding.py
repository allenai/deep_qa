from overrides import overrides
from keras.layers import Embedding


class TimeDistributedEmbedding(Embedding):
    '''
    Embedding already works for inputs of any shape. We just need to not constrain its returned output shape with
    the assumption that the input is 2D.
    '''
    @overrides
    def compute_output_shape(self, input_shape):
        if not self.input_length:
            input_length = input_shape[-1]
        else:
            input_length = self.input_length
        return input_shape[:-1] + (input_length, self.output_dim)
