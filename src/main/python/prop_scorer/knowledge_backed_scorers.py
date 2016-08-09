'''
Knowledge backed scorers take an encoded sentence (or logical form) representation
and encoded representations of background facts related to the sentence, and summarize
the background information as a weighted average of the representations of background
facts, conditioned on the encoded sentence. For example, MemoryLayer can be
used as the first layer in an MLP to make a memory network.
'''

from keras.engine import InputSpec
from keras import backend as K
from keras.layers import Dense


class MemoryLayer(Dense):
    """
    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the sentence, i[:, 1:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    Attend to facts conditioned on the input sentence, and output the sum of the sentence encoding
    and the averaged fact encoding.  This layer is a reimplementation of the memory layer in
    "End-to-End Memory Networks", Sukhbaatar et al. 2015.  Thus the attention is done with a simple
    dot product, and a sum at the end to combine the aggregated memory with the input.
    """

    def __init__(self, output_dim, **kwargs):
        # Assuming encoded knowledge and encoded input sentence are of the same dimensionality. So
        # we will not change the input_dim, and rely on the underlying Dense layer to specify it.
        kwargs['output_dim'] = output_dim
        super(MemoryLayer, self).__init__(**kwargs)
        # Now that the constructor of Dense is called, ndim will have been set to 2. Change it to
        # 3. Or else build will complain when it sees as 3D tensor as input.
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        assert len(input_shape) == 3
        dense_input_shape = (input_shape[0], input_shape[2])  # Eliminating second dim.
        super(MemoryLayer, self).build(dense_input_shape)
        # Dense's build method would have changed the input shape, and thus the ndim again.  Change
        # it back.
        self.input_spec = [InputSpec(shape=input_shape)]

    def call(self, x, mask=None):
        # Do the attention magic, transform the input to two dimensions, and pass it along to the
        # call method of Dense.
        # Assumption: The first row in each slice corresponds to the encoding of the input and the
        # remaining rows to those of the background knowledge.

        sentence_encoding = x[:, 0, :]  # (num_samples, input_dim)
        knowledge_encoding = x[:, 1:, :]  # (num_samples, knowledge_length, input_dim)

        # We want to take a dotproduct of the knowledge matrix and the sentence vector from each
        # sample. Instead of looping over all samples (inefficient), let's tile the sentence
        # encoding to make it the same size as knowledge encoding, take an element wise product and
        # sum over the last dimension (dim = 2).
        knowledge_length = knowledge_encoding.shape[1]
        tiled_sentence_encoding = K.permute_dimensions(
                K.tile(sentence_encoding, (knowledge_length, 1, 1)),
                (1, 0, 2))  # (num_samples, knowledge_length, input_dim)
        knowledge_attention = K.softmax(K.sum(knowledge_encoding * tiled_sentence_encoding,
                                              axis=2))  # (num_samples, knowledge_length)

        # Expand attention matrix to make it a tensor with last dim of length 1 so that we can do
        # an element wise multiplication with knowledge, and then sum out the knowledge dimension
        # to make it a weighted average
        attended_knowledge = K.sum(knowledge_encoding * K.expand_dims(knowledge_attention, dim=-1),
                                   axis=1)  # (num_samples, input_dim)

        # Summing the sentences and attended knowledge vectors, following the End to End Memory
        # networks paper (Sukhbaatar et al.,'15).
        dense_layer_input = sentence_encoding + attended_knowledge
        output = super(MemoryLayer, self).call(dense_layer_input)
        return output

    def get_output_shape_for(self, input_shape):
        dense_input_shape = (input_shape[0], input_shape[2],)  # Eliminating second dim.
        return super(MemoryLayer, self).get_output_shape_for(dense_input_shape)


class AttentiveReaderLayer(Dense):
    """
    This is very similar to the MemoryLayer, but uses different equations to combine the background
    "memory" and the input.  Here we are reimplementing the central memory layer described in
    "Teaching Machines to Read and Comprehend", Hermann et al., 2015.

    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the sentence, i[:, 1:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    This layer concatenates the input with each background sentence, passes them through a
    non-linearity, then does a softmax to get attention weights.  After the aggregate memory is
    constructed, we again concatenate it with the input and pass it through a non-linearity to get
    a final output from this layer.

    Equations:
    Inputs: u is the sentence encoding, z_t are the background sentence encodings
    Weights: W_1 (called self.dense_weights), v (called self.dot_bias), W_2
    Output: y

    m_t = tanh(W_1 * concat(z_t, u))
    q_t = dot(v, m_t)
    a_t = softmax(q_t)
    r = sum_t(a_t * z_t)
    y = tanh(W_2 * concat(r, u))
    """

    def __init__(self, output_dim, **kwargs):
        # Assuming encoded knowledge and encoded input sentence are of the same dimensionality. So
        # we will not change the input_dim, and rely on the underlying Dense layer to specify it.
        kwargs['output_dim'] = output_dim

        # We want to default to a tanh activation.  You can override it if you want, but if it's
        # not specified, use tanh instead of linear.
        if 'activation' not in kwargs:
            kwargs['activation'] = 'tanh'
        super(AttentiveReaderLayer, self).__init__(**kwargs)

        # Now that the constructor of Dense is called, ndim will have been set to 2. Change it to
        # 3. Or else build will complain when it sees as 3D tensor as input.
        self.input_spec = [InputSpec(ndim=3)]

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init,invalid-name
        '''
        Equations repeated from above:
        Inputs: u is the sentence encoding, z_t are the background sentence encodings
        Weights: W_1 (called self.dense_weights), v (called self.dot_bias), W_2
        Output: y

        m_t = tanh(W_1 * concat(z_t, u))
        q_t = dot(v, m_t)
        a_t = softmax(q_t)
        r = sum_t(a_t * z_t)
        y = tanh(W_2 * concat(r, u))

        Because we're subclassing Dense, the Dense layer will handle W_2 and the final tanh with
        its own weights.  So we need to build W_1 and v here.

        TODO(matt): there are a bunch of things that Keras's Dense layer has that we don't
        implement here yet: regularizers, biases, etc.  We'll not worry about those for now, but
        might want to in the future.  But note that Keras adds a bias by default, which will go in
        the computation of y, and any regularizer passed into __init__ will apply to W2.
        '''
        # If any initial weights were passed to the constructor, we have to set them here. Keras's
        # build methods need to be called so that all the trainable weight variables get
        # initialzed.  However, those methods don't play nicely with subclasses that have different
        # weight structures.  So we hide the initial weights here, so Keras ignores them, then call
        # set_weights() ourselves at the end of the method.
        self.initial_attentive_reader_weights = self.initial_weights
        self.initial_weights = None

        # The input to the dense layer will only have two dims, and will take a concatenated input,
        # so we double the length of the second dimension.
        assert len(input_shape) == 3
        dense_input_shape = (input_shape[0], input_shape[2] * 2)
        super(AttentiveReaderLayer, self).build(dense_input_shape)

        # Dense's build method would have changed the input shape, and thus the ndim again.  Here
        # we change it back.
        self.input_spec = [InputSpec(shape=input_shape)]

        input_dim = input_shape[2]
        self.dense_weights = self.init((input_dim * 2, input_dim), name='{}_inner_dense'.format(self.name))
        self.dot_bias = self.init((input_dim, 1), name='{}_inner_dot_bias'.format(self.name))
        self.trainable_weights.extend([self.dense_weights, self.dot_bias])

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_attentive_reader_weights is not None:
            self.set_weights(self.initial_attentive_reader_weights)
            del self.initial_attentive_reader_weights

    def call(self, x, mask=None):
        '''
        Equations repeated from above (last time):
        Inputs: u is the sentence encoding, z_t are the background sentence encodings
        Weights: W_1 (called self.dense_weights), v (called self.dot_bias), W_2
        Output: y

        (1) zu_t = concat(z_t, u)
        (2) m_t = tanh(dot(W_1, zu_t))
        (3) q_t = dot(v, m_t)
        (4) a_t = softmax(q_t)
        (5) r = sum_t(a_t * z_t)
        (6) y = tanh(W_2 * concat(r, u))

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.  Also, recall from above that we're leaving the last equation
        to the Dense layer that we're subclassing.  So this method does the math for the attention
        specified in the earlier equations, then passes the resultant vector off to super.call().
        '''
        # Remember that the first row in each slice corresponds to the encoding of the input and
        # the remaining rows to those of the background knowledge.
        sentence_encoding = x[:, 0, :]  # (num_samples, input_dim)
        knowledge_encoding = x[:, 1:, :]  # (num_samples, knowledge_length, input_dim)

        # We're going to have to do several operations on the input sentence for each background
        # sentence.  Instead of looping over the background sentences, which is inefficient, we'll
        # tile the sentence encoding and use it in what follows.
        knowledge_length = knowledge_encoding.shape[1]
        tiled_sentence_encoding = K.permute_dimensions(
                K.tile(sentence_encoding, (knowledge_length, 1, 1)),
                (1, 0, 2))  # (num_samples, knowledge_length, input_dim)

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 2)
        concatenated_encodings = K.concatenate([knowledge_encoding, tiled_sentence_encoding])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings, self.dense_weights))

        # (3: q_t) Result of this is (num_samples, knowledge_length).  We need to remove a dimension
        # after the dot product with K.squeeze, otherwise this would be (num_samples,
        # knowledge_length, 1), which is not a valid input to K.softmax.
        unnormalized_attention = K.squeeze(K.dot(concatenated_activation, self.dot_bias), axis=2)

        # (4: a_t) Result is (num_samples, knowledge_length)
        knowledge_attention = K.softmax(unnormalized_attention)

        # Here we expand the attention matrix to make it a tensor with last dim of length 1 so that
        # we can do an element wise multiplication with knowledge, and then we sum out the
        # knowledge dimension to make it a weighted average.
        # (5: r) Result is (num_samples, input_dim)
        attended_knowledge = K.sum(knowledge_encoding * K.expand_dims(knowledge_attention, dim=-1), axis=1)

        # Finally, we concatenate the attended knowledge vector with the input sentence encoding,
        # and pass it to the dense layer.
        # (6: y) Output is (num_samples, input_dim)
        dense_layer_input = K.concatenate([attended_knowledge, sentence_encoding])
        output = super(AttentiveReaderLayer, self).call(dense_layer_input)
        return output

    def get_output_shape_for(self, input_shape):
        dense_input_shape = (input_shape[0], input_shape[2],)  # Eliminating second dim.
        return super(AttentiveReaderLayer, self).get_output_shape_for(dense_input_shape)
