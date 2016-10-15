'''
Knowledge selectors take an encoded sentence (or logical form) representation and encoded
representations of background facts related to the sentence, and compute an attention over the
background representations. By default, the attention is soft (the attention values are in the
range (0, 1)). But we can optionally pass 'hard_selection=True' to the constructor, to make it
hard (values will be all 0, except one).
'''

from collections import OrderedDict

from keras.engine import InputSpec
from keras import backend as K
from keras import activations, initializations
from keras.layers import Layer


def tile_vector(vector, matrix):
    """
    This method takes a (collection of) vector(s) (shape: (batch_size, vector_dim)), and tiles that
    vector a number of times, giving a matrix of shape (batch_size, tile_length, vector_dim).  (I
    say "vector" and "matrix" here because I'm ignoring the batch_size).  We need the matrix as
    input so we know what the tile_length is - the matrix is otherwise ignored.

    This is necessary in a number of places in the code.  For instance, if you want to do a dot
    product of a vector with all of the vectors in a matrix, the most efficient way to do that is
    to tile the vector first, then do an element-wise product with the matrix, then sum out the
    last mode.  So, we capture this functionality here.

    This is not done as a Keras Layer, however; if you want to use this function, you'll need to do
    it _inside_ of a Layer somehow, either in a Lambda or in the call() method of a Layer you're
    writing.

    TODO(matt): it probably would make sense to move this to a central place, maybe under common
    (make a new common.tensors?).
    """
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (tile_length, batch_size, vector_dim)
    k_ones = K.permute_dimensions(K.ones_like(matrix), [1, 0, 2])

    # Now we have a (tile_length, batch_size, vector_dim)*(num_samples, vector_dim)
    # elementwise multiplication which is broadcast. We then reshape back.
    tiled_vector = K.permute_dimensions(k_ones * vector, [1, 0, 2])
    return tiled_vector


def split_selector_inputs(inputs):
    """
    To make it easier to TimeDistribute the memory computation, we smash the current memory, the
    original question, and the background knowledge into a single tensor before passing them to the
    knowledge selectors.  Here we split them out for you.  We assume that you've put the question
    first, then the current memory, then all of the background.
    """
    question_encoding = inputs[:, 0, :]
    memory_encoding = inputs[:, 1, :]
    knowledge_encoding = inputs[:, 2:, :]
    return question_encoding, memory_encoding, knowledge_encoding


def hardmax(unnormalized_attention, knowledge_length):
    # (num_samples, knowledge_length)
    tiled_max_values = K.tile(K.expand_dims(K.max(unnormalized_attention, axis=1)), (1, knowledge_length))
    # We now have a matrix where every column in each row has the max knowledge score value from
    # the corresponding row in the unnormalized attention matrix.  Next, we will compare that
    # all-max matrix with the original input, resulting in ones where the column equals max and
    # zero everywhere else.
    # Shape: (num_samples, knowledge_length)
    bool_max_attention = K.equal(unnormalized_attention, tiled_max_values)
    # Needs to be cast to be compatible with TensorFlow
    max_attention = K.cast(bool_max_attention, 'float32')
    return max_attention


class DotProductKnowledgeSelector(Layer):
    """
    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the original sentence,
    i[:, 1, :] is the encoding of the memory representation and i[:, 2:, :] are the encodings of
    the background facts. There is no need to specify knowledge length here.

    Attend to facts conditioned on the memory vector, just using a dot product between the memory
    vector and the background vectors (i.e., there are no parameters here).  This layer is a
    reimplementation of the memory layer in "End-to-End Memory Networks", Sukhbaatar et al. 2015.
    """
    def __init__(self, hard_selection=False, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]
        self.hard_selection = hard_selection
        super(DotProductKnowledgeSelector, self).__init__(**kwargs)

    def call(self, x, mask=None):
        _, memory_encoding, knowledge_encoding = split_selector_inputs(x)

        # We want to take a dot product of the knowledge matrix and the sentence vector from each
        # sample. Instead of looping over all samples (inefficient), let's tile the sentence
        # encoding to make it the same size as knowledge encoding, take an element wise product and
        # sum over the last dimension (dim = 2).

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)

        # (num_samples, knowledge_length)
        unnormalized_attention = K.sum(knowledge_encoding * tiled_memory_encoding, axis=2)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = K.softmax(unnormalized_attention)
        return knowledge_attention

    def get_output_shape_for(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)


class ParameterizedKnowledgeSelector(Layer):
    """
    Here we are reimplementing the attention part of the memory layer described in
    "Teaching Machines to Read and Comprehend", Hermann et al., 2015.

    Input Shape: num_samples, (knowledge_length + 1), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the original question,
    i[:, 1, :] is the encoding of the memory representation and i[:, 2:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    This layer concatenates the memory vector with each background sentence, passes them through a
    non-linearity, then does a softmax to get attention weights.

    Equations:
    Inputs: u is the sentence encoding, z_t are the background sentence encodings
    Weights: W_1 (called self.dense_weights), v (called self.dot_bias)
    Output: a_t

    m_t = tanh(W_1 * concat(z_t, u))
    q_t = dot(v, m_t)
    a_t = softmax(q_t)
    """

    def __init__(self,
                 activation='tanh',
                 initialization='glorot_uniform',
                 hard_selection=False,
                 weights=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.init = initializations.get(initialization)
        self.hard_selection = hard_selection
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.dense_weights = None
        self.dot_bias = None
        super(ParameterizedKnowledgeSelector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.dense_weights = self.init((input_dim * 2, input_dim), name='{}_dense'.format(self.name))
        self.dot_bias = self.init((input_dim, 1), name='{}_dot_bias'.format(self.name))
        self.trainable_weights = [self.dense_weights, self.dot_bias]

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        '''
        Equations repeated from above:
        Inputs: u is the sentence encoding, z_t are the background sentence encodings
        Weights: W_1 (called self.dense_weights), v (called self.dot_bias)
        Output: a_t

        (1) zu_t = concat(z_t, u)
        (2) m_t = tanh(dot(W_1, zu_t))
        (3) q_t = dot(v, m_t)
        (4) a_t = softmax(q_t)

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.
        '''
        _, memory_encoding, knowledge_encoding = split_selector_inputs(x)

        # We're going to have to do several operations on the memory representation for each
        # background sentence.  Instead of looping over the background sentences, which is
        # inefficient, we'll tile the sentence encoding and use it in what follows.

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 2)
        concatenated_encodings = K.concatenate([knowledge_encoding, tiled_memory_encoding])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings, self.dense_weights))

        # (3: q_t) Result of this is (num_samples, knowledge_length).  We need to remove a dimension
        # after the dot product with K.squeeze, otherwise this would be (num_samples,
        # knowledge_length, 1), which is not a valid input to K.softmax.
        unnormalized_attention = K.squeeze(K.dot(concatenated_activation, self.dot_bias), axis=2)

        # (4: a_t) Result is (num_samples, knowledge_length)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = K.softmax(unnormalized_attention)
        return knowledge_attention

    def get_output_shape_for(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)


class ParameterizedHeuristicMatchingKnowledgeSelector(Layer):
    '''
    A ParameterizedHeuristicMatchingKnowledgeSelector implements a very similar mechanism as the
    ParameterizedKnowledge Selector for attending over background knowledge. The only differences
    are:

    - The input query z_t is made up of various interactions between the original question
      representation, the memory representation and the background knowledge, rather than just
      concatenating the memory and background knowledge together.

    - Two bias vectors, b_1 and b_2 are introduced in addition to the dot_bias and dense_weights.

    This implementation follows the 'Eposodic Memory Module' in Dynamic Memory Networks for Visual
    and Textual Question Answering (page 4): https://arxiv.org/pdf/1603.01417v1.pdf
    '''
    def __init__(self,
                 activation='tanh',
                 initialization='glorot_uniform',
                 hard_selection=False,
                 weights=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.init = initializations.get(initialization)
        self.hard_selection = hard_selection
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.dense_weights = None
        self.dot_bias = None
        self.bias1 = None
        self.bias2 = None
        super(ParameterizedHeuristicMatchingKnowledgeSelector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.dense_weights = self.init((input_dim * 4, input_dim), name='{}_dense'.format(self.name))
        self.dot_bias = self.init((input_dim, 1), name='{}_dot_bias'.format(self.name))
        self.bias1 = self.init((input_dim,), name='{}_dense_bias1'.format(self.name))
        self.bias2 = self.init((1,), name='{}_dense_bias2'.format(self.name))
        self.trainable_weights = [self.dense_weights, self.dot_bias, self.bias1, self.bias2]

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        '''
        Equations repeated from above:
        Inputs: u is the sentence encoding, y is the memory encoding, z_t are the background sentence encodings
        Weights: W_1 (called self.dense_weights), v (called self.dot_bias) and b_1, b_2 are bias vectors.
        Output: a_t

        (1) zu_t = concat(z_t*u, z_t*y, |z_t - u|, |z_t - y|)
        (2) m_t = tanh(dot(W_1, zu_t) + b_1)
        (3) q_t = dot(v, m_t) + b_2
        (4) a_t = softmax(q_t)

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.
        '''
        original_question_encoding, memory_encoding, knowledge_encoding = split_selector_inputs(x)

        # We're going to have to do several operations on the question and memory representations
        # for each background sentence.  Instead of looping over the background sentences, which
        # is inefficient, we'll tile the question and memory encodings and use them in what follows.

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)
        tiled_question_encoding = tile_vector(original_question_encoding, knowledge_encoding)

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 4)
        concatenated_encodings = K.concatenate([knowledge_encoding * tiled_question_encoding,
                                                knowledge_encoding * tiled_memory_encoding,
                                                K.abs(knowledge_encoding - tiled_question_encoding),
                                                K.abs(knowledge_encoding - tiled_memory_encoding)])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings,
                                                        self.dense_weights) + self.bias1)

        # (3: q_t) Result of this is (num_samples, knowledge_length).  We need to remove a dimension
        # after the dot product with K.squeeze, otherwise this would be (num_samples,
        # knowledge_length, 1), which is not a valid input to K.softmax.
        unnormalized_attention = K.squeeze(K.dot(concatenated_activation,
                                                 self.dot_bias) + self.bias2, axis=2)

        # (4: a_t) Result is (num_samples, knowledge_length)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = K.softmax(unnormalized_attention)
        return knowledge_attention

    def get_output_shape_for(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)


# The first item added here will be used as the default in some cases.
selectors = OrderedDict()  # pylint: disable=invalid-name
selectors['parameterized'] = ParameterizedKnowledgeSelector
selectors['dot_product'] = DotProductKnowledgeSelector
selectors['parameterized_heuristic_matching'] = ParameterizedHeuristicMatchingKnowledgeSelector
