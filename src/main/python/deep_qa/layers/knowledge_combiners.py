'''
Knowledge combiners take:
 - encoded representations of background facts related to the sentence
 - attention weights over the background
 as a single tensor.

 These are then combined in some way to return a single representation of the
 background knowledge per sample. The simplest way for this to happen is simply
 taking a weighted average of the knowledge representations with respect to the
 attention weights.

 Input shapes: (samples, knowledge_len, input_dim + 1)
 Output shape: (samples, input_dim)
'''

from collections import OrderedDict
from overrides import overrides
import numpy as np

from keras.engine import InputSpec
from keras import backend as K
from keras.layers.recurrent import GRU, time_distributed_dense
from keras.layers import Layer


class WeightedAverageKnowledgeCombiner(Layer):
    '''
    A WeightedAverageKnowledgeCombiner takes a tensor formed by prepending an attention mask
    onto an encoded representation of background knowledge. Here, we simply split off the
    attention mask and use it to take a weighted average of the background vectors.
    '''

    def __init__(self, **kwargs):
        self.input_spec = [InputSpec(ndim=3)]
        self.name = kwargs.pop('name')
        # These parameters are passed for consistency with the
        # AttentiveGRUKnowlegeCombiner. They are not used here.
        kwargs.pop('output_dim')
        kwargs.pop('input_length')
        super(WeightedAverageKnowledgeCombiner, self).__init__(**kwargs)

    def call(self, x, mask=None):
        attention = x[:, :, 0]  # (samples, knowledge_length)
        x = x[:, :, 1:]  # (samples, knowledge_length, word_dim)
        return K.sum(K.expand_dims(attention, 2) * x, 1)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[2] - 1)


class AttentiveGRUKnowledgeCombiner(GRU):
    '''
    GRUs typically operate over sequences of words. Here we are are operating over the background knowledge
    sentence representations as though they are a sequence (i.e. each background sentence has already
    been encoded into a single sentence representation). The motivation behind this encoding is that
    a weighted average loses ordering information in the background knowledge - for instance, this is
    important in the BABI tasks.

    See Dynamic Memory Networks for more information: https://arxiv.org/pdf/1603.01417v1.pdf.

    This class extends the Keras Gated Recurrent Unit by implementing a method which substitutes
    the GRU update gate (normally a vector, z - it is noted below where it is normally computed) for a scalar
    attention weight (one per input, such as from the output of a softmax over the input vectors), which is
    pre-computed. As mentioned above, instead of using word embedding sequences as input to the GRU,
    we are using sentence encoding sequences.

    The implementation of this class is subtle - it is only very slightly different from a standard GRU.
    When it is initialised, the Keras backend will call the build method. It uses this to check that inputs being
    passed to this function are the correct size, so we allow this to be the actual input size as normal.

    However, for the internal implementation, everywhere where this global shape is used, we override it to be one
    less, as we are passing in a tensor of shape (batch, knowledge_length, 1 + encoding_dim) as we are including
    the attention mask. Therefore, we need all of the weights to have shape (*, encoding_dim),
    NOT (*, 1 + encoding_dim). All of the below methods which are overridden use some
    form of this dimension, so we correct them.
    '''

    def __init__(self, output_dim, input_length, **kwargs):
        self.name = kwargs.pop('name')
        super(AttentiveGRUKnowledgeCombiner, self).__init__(output_dim,
                                                            input_length=input_length,
                                                            input_dim=output_dim + 1,
                                                            name=self.name, **kwargs)

    @overrides
    def step(self, x, states):
        # pylint: disable=invalid-name
        '''
        The input to step is a tensor of shape (batch, 1 + encoding_dim), i.e. a timeslice of
        the input to this AttentiveGRU, where the time axis is the knowledge_length.
        Before we start, we strip off the attention from the beginning. Then we do the equations for a
        normal GRU, except we don't calculate the output gate z, substituting the attention weight for
        it instead.

        Note that there is some redundancy here - for instance, in the GPU mode, we do a
        larger matrix multiplication than required, as we don't use one part of it. However, for
        readability and similarity to the original GRU code in Keras, it has not been changed. In each section,
        there are commented out lines which contain code. If you were to uncomment these, remove the differences
        in the input size and replace the attention with the z gate at the output, you would have a standard
        GRU back again. We literally copied the Keras GRU code here, making some small modifications.
        '''
        attention = x[:, 0]
        x = x[:, 1:]
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.consume_less == 'gpu':

            matrix_x = K.dot(x * B_W[0], self.W) + self.b
            matrix_inner = K.dot(h_tm1 * B_U[0], self.U[:, :2 * self.output_dim])

            x_r = matrix_x[:, self.output_dim: 2 * self.output_dim]
            inner_r = matrix_inner[:, self.output_dim: 2 * self.output_dim]
            # x_z = matrix_x[:, :self.output_dim]
            # inner_z = matrix_inner[:, :self.output_dim]

            # z = self.inner_activation(x_z + inner_z)
            r = self.inner_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.output_dim:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.U[:, 2 * self.output_dim:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.consume_less == 'cpu':
                # x_z = x[:, :self.output_dim]
                x_r = x[:, self.output_dim: 2 * self.output_dim]
                x_h = x[:, 2 * self.output_dim:]
            elif self.consume_less == 'mem':
                # x_z = K.dot(x * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(x * B_W[1], self.W_r) + self.b_r
                x_h = K.dot(x * B_W[2], self.W_h) + self.b_h
            else:
                raise Exception('Unknown `consume_less` mode.')

            # z = self.inner_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.inner_activation(x_r + K.dot(h_tm1 * B_U[1], self.U_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.U_h))

        # Here is the KEY difference between a GRU and an AttentiveGRU. Instead of using
        # a learnt output gate (z), we use a scalar attention vector (batch, 1) for this
        # particular background knowledge vector.
        h = K.expand_dims(attention, 1) * hh + (1 - K.expand_dims(attention, 1)) * h_tm1
        return h, [h]

    @overrides
    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        '''
        This is used by Keras to verify things, but also to build the weights.
        The only differences from the Keras GRU (which we copied exactly
        other than the below) are:
        - We generate weights with dimension input_dim[2] - 1, rather than
          dimension input_dim[2].
        - There are a few variables which are created in non-'gpu' modes which
          are not required, and actually raise errors in Theano if you include them in
          the trainable weights(as Theano will alert you if you try to compute a gradient
          of a loss wrt a constant). These are commented out but left in for clarity below.
        '''
        self.input_spec = [InputSpec(shape=input_shape)]

        # Here we make all the weights with a dimension one smaller
        # than the input, as we remove the attention weights.
        self.input_dim = input_shape[2] - 1

        if self.stateful:
            self.reset_states()
        else:
            # initial states: all-zero tensor of shape (output_dim)
            self.states = [None]

        if self.consume_less == 'gpu':

            self.W = self.init((self.input_dim, 3 * self.output_dim),
                               name='{}_W'.format(self.name))
            self.U = self.inner_init((self.output_dim, 3 * self.output_dim),
                                     name='{}_U'.format(self.name))

            self.b = K.variable(np.hstack((np.zeros(self.output_dim),
                                           np.zeros(self.output_dim),
                                           np.zeros(self.output_dim))),
                                name='{}_b'.format(self.name))

            self.trainable_weights = [self.W, self.U, self.b]
        else:

            self.W_z = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_z'.format(self.name))
            self.U_z = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_z'.format(self.name))
            self.b_z = K.zeros((self.output_dim,), name='{}_b_z'.format(self.name))

            self.W_r = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_r'.format(self.name))
            self.U_r = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_r'.format(self.name))
            self.b_r = K.zeros((self.output_dim,), name='{}_b_r'.format(self.name))

            self.W_h = self.init((self.input_dim, self.output_dim),
                                 name='{}_W_h'.format(self.name))
            self.U_h = self.inner_init((self.output_dim, self.output_dim),
                                       name='{}_U_h'.format(self.name))
            self.b_h = K.zeros((self.output_dim,), name='{}_b_h'.format(self.name))

            # self.W_z, self.U_z, self.b_z - we don't need these parameters anymore.
            self.trainable_weights = [self.W_r, self.U_r, self.b_r,
                                      self.W_h, self.U_h, self.b_h]

            self.W = K.concatenate([self.W_z, self.W_r, self.W_h])
            self.U = K.concatenate([self.U_z, self.U_r, self.U_h])
            self.b = K.concatenate([self.b_z, self.b_r, self.b_h])

        self.regularizers = []
        if self.W_regularizer:
            self.W_regularizer.set_param(self.W)
            self.regularizers.append(self.W_regularizer)
        if self.U_regularizer:
            self.U_regularizer.set_param(self.U)
            self.regularizers.append(self.U_regularizer)
        if self.b_regularizer:
            self.b_regularizer.set_param(self.b)
            self.regularizers.append(self.b_regularizer)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True
    @overrides
    def preprocess_input(self, x):
        '''
        We have to override this preprocessing step, because if we are using the cpu,
        we do the weight - input multiplications in the internals of the GRU as seperate,
        smaller matrix multiplications and concatenate them after. Therefore, before this
        happens, we split off the attention and then add it back afterwards.
        '''
        if self.consume_less == 'cpu':

            attention = x[:, :, 0]  # Shape:(samples, knowledge_length)
            x = x[:, :, 1:]  # Shape:(samples, knowledge_length, word_dim)

            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2] - 1
            timesteps = input_shape[1]

            x_z = time_distributed_dense(x, self.W_z, self.b_z, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_r = time_distributed_dense(x, self.W_r, self.b_r, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)
            x_h = time_distributed_dense(x, self.W_h, self.b_h, self.dropout_W,
                                         input_dim, self.output_dim, timesteps)

            # Add attention back on to it's original place.
            return K.concatenate([K.expand_dims(attention, 2), x_z, x_r, x_h], axis=2)
        else:
            return x

# The first item added here will be used as the default in some cases.
knowledge_combiners = OrderedDict()  # pylint:  disable=invalid-name
knowledge_combiners["weighted_average"] = WeightedAverageKnowledgeCombiner
knowledge_combiners["attentive_gru"] = AttentiveGRUKnowledgeCombiner
