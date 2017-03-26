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
from keras import activations

from ..tensors.backend import tile_vector, hardmax
from ..tensors.masked_operations import masked_softmax
from .masked_layer import MaskedLayer


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

def split_selector_masks(inputs):
    """
    This function is used in the same context as split_selector_inputs, but for masks, which are
    a dimension smaller.
    """
    question_mask = inputs[:, 0]
    memory_mask = inputs[:, 1]
    knowledge_mask = inputs[:, 2:]
    return question_mask, memory_mask, knowledge_mask

class DotProductKnowledgeSelector(MaskedLayer):
    """
    Input Shape: num_samples, (knowledge_length + 2), input_dim

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

    def call(self, inputs, mask=None):
        _, memory_encoding, knowledge_encoding = split_selector_inputs(inputs)

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)
        if mask is not None:
            # This mask is (samples, knowledge_length), so we need to expand it to multiply with the actual
            # knowledge_encoding. At this point, there is no mask for the question or memory encoding.
            _, _, mask = split_selector_masks(mask)
            knowledge_mask = K.expand_dims(K.cast(mask, 'float32'))
            knowledge_encoding *= knowledge_mask
        # (num_samples, knowledge_length)
        unnormalized_attention = K.sum(knowledge_encoding * tiled_memory_encoding, axis=2)

        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = masked_softmax(unnormalized_attention, mask)
        return knowledge_attention

    def compute_output_shape(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)

    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        # At this point, the attention already implicitly contains our mask. The other
        # inputs to the Knowledge Combiners will still have access to their respective masks,
        # so we don't need to pass on the dimensions which were masked here as well, as they
        # are already zeroed out.
        return None


class ParameterizedKnowledgeSelector(MaskedLayer):
    """
    Here we are reimplementing the attention part of the memory layer described in
    "Teaching Machines to Read and Comprehend", Hermann et al., 2015.

    Input Shape: num_samples, (knowledge_length + 2), input_dim

    Take the input as a tensor i, such that i[:, 0, :] is the encoding of the original question,
    i[:, 1, :] is the encoding of the memory representation and i[:, 2:, :]
    are the encodings of the background facts.  There is no need to specify knowledge length here.

    This layer concatenates the memory vector with each background sentence, passes them through a
    non-linearity, then does a softmax to get attention weights.

    Inputs:
        - A sentence encoding :math:`u`, with shape ``(batch_size, encoding_dim)``
        - Background sentence encodings :math:`z_t` with shape ``(batch_size, knowledge_length, encoding_dim)``

    Weights:
        - :math:`W_1` (called ``self.dense_weights``)
        - :math:`v` (called ``self.dot_bias``)

    Output:
        - :math:`a_t`

    Equations:
        - :math:`m_t = tanh(W_1 * concat(z_t, u))`
        - :math:`q_t = dot(v, m_t)`
        - :math:`a_t = softmax(q_t)`
    """

    def __init__(self,
                 activation='tanh',
                 initialization='glorot_uniform',
                 hard_selection=False,
                 weights=None,
                 **kwargs):
        self.activation = activations.get(activation)
        self.init = initialization
        self.hard_selection = hard_selection
        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        self.dense_weights = None
        self.dot_bias = None
        super(ParameterizedKnowledgeSelector, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        input_dim = input_shape[2]
        self.dense_weights = self.add_weight((input_dim * 2, input_dim),
                                             initializer=self.init,
                                             name='{}_dense'.format(self.name))
        self.dot_bias = self.add_weight((input_dim, 1),
                                        initializer=self.init,
                                        name='{}_dot_bias'.format(self.name))
        self.trainable_weights = [self.dense_weights, self.dot_bias]

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(ParameterizedKnowledgeSelector, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        Inputs:
            - A sentence encoding :math:`u`, with shape ``(batch_size, encoding_dim)``
            - Background sentence encodings :math:`z_t` with shape ``(batch_size, knowledge_length, encoding_dim)``

        Weights:
            - :math:`W_1` (called ``self.dense_weights``)
            - :math:`v` (called ``self.dot_bias``)

        Output:
            - :math:`a_t`

        Equations:
            - 1. :math:`zu_t = concat(z_t, u)`
            - 2. :math:`m_t = tanh(dot(W_1, zu_t))`
            - 3. :math:`q_t = dot(v, m_t)`
            - 4. :math:`a_t = softmax(q_t)`

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.
        '''
        _, memory_encoding, knowledge_encoding = split_selector_inputs(inputs)

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)
        if mask is not None:
            # This mask is (samples, knowledge_length), so we need to expand it to multiply with the actual
            # knowledge_encoding. At this point, there is no mask for the question or memory encoding.
            _, _, mask = split_selector_masks(mask)
            knowledge_mask = K.expand_dims(K.cast(mask, 'float32'))
            knowledge_encoding *= knowledge_mask
            tiled_memory_encoding *= knowledge_mask

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 2)
        concatenated_encodings = K.concatenate([knowledge_encoding, tiled_memory_encoding])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings, self.dense_weights))

        if mask is not None:
            concatenated_activation *= knowledge_mask

        # (3: q_t) Result of this is (num_samples, knowledge_length).  We need to remove a dimension
        # after the dot product with K.squeeze, otherwise this would be (num_samples,
        # knowledge_length, 1), which is not a valid input to K.softmax.
        unnormalized_attention = K.squeeze(K.dot(concatenated_activation, self.dot_bias), axis=2)

        # (4: a_t) Result is (num_samples, knowledge_length)
        if self.hard_selection:
            knowledge_length = K.shape(knowledge_encoding)[1]
            knowledge_attention = hardmax(unnormalized_attention, knowledge_length)
        else:
            knowledge_attention = masked_softmax(unnormalized_attention, mask)
        return knowledge_attention

    def compute_output_shape(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)

    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        # At this point, the attention already implicitly contains our mask. The other
        # inputs to the Knowledge Combiners will still have access to their respective masks,
        # so we don't need to pass on the dimensions which were masked here as well, as they
        # are already zeroed out.
        return None


class ParameterizedHeuristicMatchingKnowledgeSelector(MaskedLayer):
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
        self.init = initialization
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
        self.dense_weights = self.add_weight((input_dim * 4, input_dim),
                                             initializer=self.init,
                                             name='{}_dense'.format(self.name))
        self.dot_bias = self.add_weight((input_dim, 1),
                                        initializer=self.init,
                                        name='{}_dot_bias'.format(self.name))
        self.bias1 = self.add_weight((input_dim,),
                                     initializer=self.init,
                                     name='{}_dense_bias1'.format(self.name))
        self.bias2 = self.add_weight((1,),
                                     initializer=self.init,
                                     name='{}_dense_bias2'.format(self.name))
        self.trainable_weights = [self.dense_weights, self.dot_bias, self.bias1, self.bias2]

        # Now that trainable_weights is complete, we set weights if needed.
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        super(ParameterizedHeuristicMatchingKnowledgeSelector, self).build(input_shape)

    def call(self, inputs, mask=None):
        '''
        Equations repeated from above.

        Inputs:
            - A sentence encoding :math:`u`, with shape ``(batch_size, encoding_dim)``
            - A memory encoding :math:`y`, with shape ``(batch_size, encoding_dim)``
            - Background sentence encodings :math:`z_t` with shape ``(batch_size, knowledge_length, encoding_dim)``

        Weights:
            - :math:`W_1` (called ``self.dense_weights``)
            - :math:`v` (called ``self.dot_bias``)
            - :math:`b_1`, :math:`b_2` are bias vectors.

        Output:
            - :math`a_t`

        Equations:
            - 1. :math:`zu_t = concat(z_t*u, z_t*y, |z_t - u|, |z_t - y|)`
            - 2. :math:`m_t = tanh(dot(W_1, zu_t) + b_1)`
            - 3. :math:`q_t = dot(v, m_t) + b_2`
            - 4. :math:`a_t = softmax(q_t)`

        Here we actually implement the logic of these equations.  We label each step with its
        number and the variable above that it's computing.  The implementation looks more complex
        than these equations because we have to unpack the input, then use tiling instead of loops
        to make this more efficient.
        '''
        original_question_encoding, memory_encoding, knowledge_encoding = split_selector_inputs(inputs)

        # (num_samples, knowledge_length, input_dim)
        tiled_memory_encoding = tile_vector(memory_encoding, knowledge_encoding)
        tiled_question_encoding = tile_vector(original_question_encoding, knowledge_encoding)

        if mask is not None:
            # This mask is (samples, knowledge_length), so we need to expand it to multiply with the actual
            # knowledge_encoding. At this point, there is no mask for the question or memory encoding.
            _, _, mask = split_selector_masks(mask)
            knowledge_mask = K.expand_dims(K.cast(mask, 'float32'))

            tiled_memory_encoding *= knowledge_mask
            tiled_question_encoding *= knowledge_mask
            knowledge_encoding *= knowledge_mask

        # (1: zu_t) Result of this is (num_samples, knowledge_length, input_dim * 4)
        concatenated_encodings = K.concatenate([knowledge_encoding * tiled_question_encoding,
                                                knowledge_encoding * tiled_memory_encoding,
                                                K.abs(knowledge_encoding - tiled_question_encoding),
                                                K.abs(knowledge_encoding - tiled_memory_encoding)])

        # (2: m_t) Result of this is (num_samples, knowledge_length, input_dim)
        concatenated_activation = self.activation(K.dot(concatenated_encodings, self.dense_weights) + self.bias1)

        if mask is not None:
            concatenated_activation *= knowledge_mask

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
            knowledge_attention = masked_softmax(unnormalized_attention, mask)
        return knowledge_attention

    def compute_output_shape(self, input_shape):
        # For each sample, the output is a vector of size knowledge_length, indicating the weights
        # over background information.
        return (input_shape[0], input_shape[1] - 2)  # (num_samples, knowledge_length)

    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        # At this point, the attention already implicitly contains our mask. The other
        # inputs to the Knowledge Combiners will still have access to their respective masks,
        # so we don't need to pass on the dimensions which were masked here as well, as they
        # are already zeroed out.
        return None

# The first item added here will be used as the default in some cases.
selectors = OrderedDict()  # pylint: disable=invalid-name
selectors['parameterized'] = ParameterizedKnowledgeSelector
selectors['dot_product'] = DotProductKnowledgeSelector
selectors['parameterized_heuristic_matching'] = ParameterizedHeuristicMatchingKnowledgeSelector
