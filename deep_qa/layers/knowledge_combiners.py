"""
Knowledge combiners take:

- Encoded representations of background facts related to the sentence
- Attention weights over the background as a single tensor.

These are then combined in some way to return a single representation of the
background knowledge per sample. The simplest way for this to happen is simply
taking a weighted average of the knowledge representations with respect to the
attention weights.

Input shapes:
    - (samples, knowledge_len, input_dim + 1)

Output shape:
    - (samples, input_dim)
"""

from collections import OrderedDict
from overrides import overrides

from keras.engine import InputSpec
from keras import backend as K
from keras.layers import Layer
from keras.layers.recurrent import GRU, _time_distributed_dense


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

    @overrides
    def call(self, inputs):
        attention = inputs[:, :, 0]  # (samples, knowledge_length)
        inputs = inputs[:, :, 1:]  # (samples, knowledge_length, word_dim)
        return K.sum(K.expand_dims(attention, 2) * inputs, 1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2] - 1)

    @overrides
    def get_config(self):
        config = {
                "output_dim": -1,
                "input_length": -1
                }
        base_config = super(WeightedAverageKnowledgeCombiner, self).get_config()
        config.update(base_config)
        return config


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
    def step(self, inputs, states):
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
        attention = inputs[:, 0]
        inputs = inputs[:, 1:]
        h_tm1 = states[0]  # previous memory
        B_U = states[1]  # dropout matrices for recurrent units
        B_W = states[2]

        if self.implementation == 2:

            matrix_x = K.dot(inputs * B_W[0], self.kernel)
            if self.use_bias:
                matrix_x = K.bias_add(matrix_x, self.bias)
            matrix_inner = K.dot(h_tm1 * B_U[0], self.recurrent_kernel[:, :2 * self.units])

            x_r = matrix_x[:, self.units: 2 * self.units]
            inner_r = matrix_inner[:, self.units: 2 * self.units]
            # x_z = matrix_x[:, :self.units]
            # inner_z = matrix_inner[:, :self.units]

            # z = self.recurrent_activation(x_z + inner_z)
            r = self.recurrent_activation(x_r + inner_r)

            x_h = matrix_x[:, 2 * self.units:]
            inner_h = K.dot(r * h_tm1 * B_U[0], self.recurrent_kernel[:, 2 * self.units:])
            hh = self.activation(x_h + inner_h)
        else:
            if self.implementation == 0:
                # x_z = inputs[:, :self.units]
                x_r = inputs[:, self.units: 2 * self.units]
                x_h = inputs[:, 2 * self.units:]
            elif self.implementation == 1:
                # x_z = K.dot(inputs * B_W[0], self.W_z) + self.b_z
                x_r = K.dot(inputs * B_W[1], self.kernel_r)
                x_h = K.dot(inputs * B_W[2], self.kernel_h)
                if self.use_bias:
                    x_r = K.bias_add(x_r, self.bias_r)
                    x_h = K.bias_add(x_h, self.bias_h)
            else:
                raise Exception('Unknown implementation')

            # z = self.recurrent_activation(x_z + K.dot(h_tm1 * B_U[0], self.U_z))
            r = self.recurrent_activation(x_r + K.dot(h_tm1 * B_U[1], self.recurrent_kernel_r))

            hh = self.activation(x_h + K.dot(r * h_tm1 * B_U[2], self.recurrent_kernel_h))

        # Here is the KEY difference between a GRU and an AttentiveGRU. Instead of using
        # a learnt output gate (z), we use a scalar attention vector (batch, 1) for this
        # particular background knowledge vector.
        h = K.expand_dims(attention, 1) * hh + (1 - K.expand_dims(attention, 1)) * h_tm1
        return h, [h]

    @overrides
    def build(self, input_shape):
        """
        This is used by Keras to verify things, but also to build the weights.
        The only differences from the Keras GRU (which we copied exactly
        other than the below) are:

        - We generate weights with dimension input_dim[2] - 1, rather than
          dimension input_dim[2].
        - There are a few variables which are created in non-'gpu' modes which
          are not required, and actually raise errors in Theano if you include them in
          the trainable weights(as Theano will alert you if you try to compute a gradient
          of a loss wrt a constant). These are commented out but left in for clarity below.
        """
        new_input_shape = list(input_shape)
        new_input_shape[2] -= 1
        super(AttentiveGRUKnowledgeCombiner, self).build(tuple(new_input_shape))
        self.input_spec = [InputSpec(shape=input_shape)]

    @overrides
    def preprocess_input(self, inputs, training=None):
        '''
        We have to override this preprocessing step, because if we are using the cpu,
        we do the weight - input multiplications in the internals of the GRU as seperate,
        smaller matrix multiplications and concatenate them after. Therefore, before this
        happens, we split off the attention and then add it back afterwards.
        '''
        if self.implementation == 0:

            attention = inputs[:, :, 0]  # Shape:(samples, knowledge_length)
            inputs = inputs[:, :, 1:]  # Shape:(samples, knowledge_length, word_dim)

            input_shape = self.input_spec[0].shape
            input_dim = input_shape[2] - 1
            timesteps = input_shape[1]

            x_z = _time_distributed_dense(inputs, self.kernel_z, self.bias_z,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_r = _time_distributed_dense(inputs, self.kernel_r, self.bias_r,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)
            x_h = _time_distributed_dense(inputs, self.kernel_h, self.bias_h,
                                          self.dropout, input_dim, self.units,
                                          timesteps, training=training)

            # Add attention back on to it's original place.
            return K.concatenate([K.expand_dims(attention, 2), x_z, x_r, x_h], axis=2)
        else:
            return inputs

# The first item added here will be used as the default in some cases.
knowledge_combiners = OrderedDict()  # pylint: disable=invalid-name
knowledge_combiners["weighted_average"] = WeightedAverageKnowledgeCombiner
knowledge_combiners["attentive_gru"] = AttentiveGRUKnowledgeCombiner
