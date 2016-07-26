from keras import initializations, activations
from keras.engine import InputSpec
from keras import backend as K
from keras.layers import Dense
'''
Knowledge backed scorers take an encoded sentence (or logical form) representation
and encoded representations of background facts related to the sentence, and summarize
the background information as a weighted average of the representations of background
facts, conditioned on the encoded sentence. For example, KnowledgeBackedDense can be
used as the first layer in an MLP to make a memory network.
'''
class KnowledgeBackedDense(Dense):
    """
    Input Shape: num_samples, (knowledge_length + 1), input_dim
    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the sentence, 
    i[:, 1:, :] are the encodings of the background facts. 
    
    Attend to facts conditioned on the input sentence, and output
    the sum of the sentence encoding and the averaged fact encoding.
    The attention mechanism uses a simple dot product, thus not adding any more parameters
    to the model. Also, there is no need to specify knowledge length here.
    
    But this can be more complex. If using an MLP for attention, we can 
    try other operations instead of a simple dot product.
    """
    def __init__(self, output_dim, **kwargs):
        # Assuming encoded knowledge and encoded input sentence are of the same 
        # dimensionality. So we will not change the input_dim, and rely on the 
        # underlying Dense layer to specify it.
        kwargs['output_dim'] = output_dim
        super(KnowledgeBackedDense, self).__init__(**kwargs)
        # Now that the constructor of Dense is called, ndim will have been set to 2. Change
        # it to 3. Or else build will complain when it sees as 3D tensor as input.
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 3
        dense_input_shape = (input_shape[0], input_shape[2])  # Eliminating second dim.
        super(KnowledgeBackedDense, self).build(dense_input_shape)
        # Dense's build method would have changed the input shape, and thus the ndim again.
        # Change it back.
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x, mask=None):
        # Do the attention magic, transform the input to two dimensions, and pass it along
        # to the call method of Dense.
        # Assumption: The first row in each slice corresponds to the encoding of the input
        # and the remaining rows to those of the background knowledge.

        sentence_encoding = x[:, 0, :]  # (num_samples, input_dim)
        knowledge_encoding = x[:, 1:, :]  # (num_samples, knowledge_length, input_dim)

        # We want to take a dotproduct of the knowledge matrix and the sentence vector
        # from each sample. Instead of looping over all samples (inefficient), let's tile
        # the sentence encoding to make it the same size as knowledge encoding, take an 
        # element wise product and sum over the last dimension (dim = 2).
        knowledge_length = knowledge_encoding.shape[1]
        tiled_sentence_encoding = K.permute_dimensions(K.tile(sentence_encoding, 
            (knowledge_length, 1, 1)), (1, 0, 2))  # (num_samples, knowledge_length, input_dim)
        knowledge_attention = K.softmax(K.sum(knowledge_encoding * tiled_sentence_encoding, 
                axis=2)) # (num_samples, knowledge_length)

        # Expand attention matrix to make it a tensor with last dim of length 1 so that 
        # we can do an element wise multiplication with knowledge, and then sum out the
        # knowledge dimension to make it a weighted average
        attended_knowledge = K.sum(knowledge_encoding * K.expand_dims(knowledge_attention, 
            dim=-1), axis=1)  # (num_samples, input_dim)

        # Summing the sentences and attended knowledge vectors, following the End to End
        # Memory networks paper (Sukhbaatar et al.,'15).
        dense_layer_input = sentence_encoding + attended_knowledge
        output = super(KnowledgeBackedDense, self).call(dense_layer_input)
        return output

    def get_output_shape_for(self, input_shape):
        dense_input_shape = (input_shape[0], input_shape[2],)  # Eliminating second dim.
        return super(KnowledgeBackedDense, self).get_output_shape_for(dense_input_shape)

