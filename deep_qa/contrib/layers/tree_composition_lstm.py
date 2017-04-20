import copy
import warnings

from keras import backend as K
from keras import activations, regularizers
from keras.engine import InputSpec
from keras.layers import Recurrent
import numpy as np

from ...data.instances.text_classification.logical_form_instance import SHIFT_OP, REDUCE2_OP, REDUCE3_OP


class TreeCompositionLSTM(Recurrent):
    '''
    Conceptual differences from LSTM:

    (1) Tree LSTM does not differentiate between x and h, because
        tree composition is not applied at every time step (it is applied
        when the input symbol is a reduce) and when it is applied, there is no
        "current input".

    (2) Input sequences are not the ones being composed, they are operations on
        the buffer containing elements corresponding to tokens. There isn't one
        token per timestep like LSTMs.

    (3) Single vectors h and c are replaced by a stack and buffer of h and c
        corresponding to the structure processed so far.

    (4) Gates are applied on two or three elements at a time depending on the
        type of reduce. Accordingly there are two classes of gates: G_2 (two
        elements) and G_3 (three elements)

    (5) G_2 has two forget gates, for each element that can be forgotten and
        G_3 has three.

    Notes
    -----
    This is almost certainly broken.  We haven't really used this since it was written, and the
    port to Keras 2 probably broke things, and we haven't had any motivation to fix it yet.  You
    have been warned.
    '''
    # pylint: disable=invalid-name
    def __init__(self,
                 units,
                 stack_limit,
                 buffer_ops_limit,
                 initializer='glorot_uniform',
                 forget_bias_initializer='one',
                 activation='tanh',
                 inner_activation='hard_sigmoid',
                 W_regularizer=None,
                 U_regularizer=None,
                 V_regularizer=None,
                 b_regularizer=None,
                 **kwargs):
        self.stack_limit = stack_limit
        # buffer_ops_limit is the max of buffer_limit and num_ops. This needs to be one value since
        # the initial buffer state and the list of operations need to be concatenated before passing
        # them to TreeCompositionLSTM
        self.buffer_ops_limit = buffer_ops_limit
        self.output_dim = units
        self.initializer = initializer
        self.forget_bias_initializer = forget_bias_initializer
        activation = kwargs.get("activation", "tanh")
        self.activation = activations.get(activation)
        self.inner_activation = activations.get(inner_activation)
        # Make two deep copies each of W, U and b since regularizers.get() method modifes them!
        W2_regularizer = copy.deepcopy(W_regularizer)
        W3_regularizer = copy.deepcopy(W_regularizer)
        U2_regularizer = copy.deepcopy(U_regularizer)
        U3_regularizer = copy.deepcopy(U_regularizer)
        b2_regularizer = copy.deepcopy(b_regularizer)
        b3_regularizer = copy.deepcopy(b_regularizer)
        # W, U and b get two copies of each corresponding regularizer
        self.W_regularizers = [regularizers.get(W2_regularizer), regularizers.get(W3_regularizer)] \
                if W_regularizer else None
        self.U_regularizers = [regularizers.get(U2_regularizer), regularizers.get(U3_regularizer)] \
                if U_regularizer else None
        self.V_regularizer = regularizers.get(V_regularizer)
        self.b_regularizers = [regularizers.get(b2_regularizer), regularizers.get(b3_regularizer)] \
                if b_regularizer else None
        # TODO(pradeep): Ensure output_dim = input_dim - 1

        self.dropout_W = kwargs["dropout_W"] if "dropout_W" in kwargs else 0.
        self.dropout_U = kwargs["dropout_U"] if "dropout_U" in kwargs else 0.
        self.dropout_V = kwargs["dropout_V"] if "dropout_V" in kwargs else 0.
        if self.dropout_W:
            self.uses_learning_phase = True

        # Pass any remaining arguments of the constructor to the super class' constructor
        super(TreeCompositionLSTM, self).__init__(**kwargs)
        if self.stateful:
            warnings.warn("Current implementation cannot be stateful. \
                    Ignoring stateful=True", RuntimeWarning)
            self.stateful = False
        if self.return_sequences:
            warnings.warn("Current implementation cannot return sequences.\
                    Ignoring return_sequences=True", RuntimeWarning)
            self.return_sequences = False

    def get_initial_states(self, inputs):
        # The initial buffer is sent into the TreeLSTM as a part of the input.
        # i.e., inputs is a concatenation of the transitions and the initial buffer.
        # (batch_size, buffer_limit, output_dim+1)
        # We will now separate the buffer and the transitions and initialize the
        # buffer state of the TreeLSTM with the initial buffer value.
        # The rest of the input is the transitions, which we do not need now.

        # Take the buffer out.
        init_h_for_buffer = inputs[:, :, 1:]  # (batch_size, buffer_limit, output_dim)
        # Initializing all c as zeros.
        init_c_for_buffer = K.zeros_like(init_h_for_buffer)

        # Each element in the buffer is a concatenation of h and c for the corresponding
        # node
        init_buffer = K.concatenate([init_h_for_buffer, init_c_for_buffer], axis=-1)
        # We need a symbolic all zero tensor of size (samples, stack_limit, 2*output_dim) for
        # init_stack The problem is the first dim (samples) is a place holder and not an actual
        # value. So we'll use the following trick
        temp_state = K.zeros_like(inputs)  # (samples, buffer_ops_limit, input_dim)
        temp_state = K.tile(K.sum(temp_state, axis=(1, 2)),
                            (self.stack_limit, 2*self.output_dim, 1))  # (stack_limit, 2*output_dim, samples)
        init_stack = K.permute_dimensions(temp_state, (2, 0, 1))  # (samples, stack_limit, 2*output_dim)
        return [init_buffer, init_stack]

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init,redefined-variable-type
        # Defining two classes of parameters:
        # 1) predicate, one argument composition (*2_*)
        # 2) predicate, two arguments composition (*3_*)
        #
        # The naming scheme is an extension of the one used
        # in the LSTM code of Keras. W is a weight and b is a bias
        # *_i: input gate parameters
        # *_fp: predicate forget gate parameters
        # *_fa: argument forget gate parameters (one-arg only)
        # *_fa1: argument-1 forget gate parameters (two-arg only)
        # *_fa2: argument-2 forget gate parameters (two-arg only)
        # *_u: update gate parameters
        # *_o: output gate parameters
        #
        # Predicate, argument composition:
        # W2_i, W2_fp, W2_fa, W2_o, W2_u
        # U2_i, U2_fp, U2_fa, U2_o, U2_u
        # b2_i, b2_fp, b2_fa, b2_o, b2_u
        #
        # Predicate, two argument composition:
        # W3_i, W3_fp, W3_fa1, W3_fa2, W3_o, W3_u
        # U3_i, U3_fp, U3_fa1, U3_fa2, U3_o, U3_u
        # V3_i, V3_fp, V3_fa1, V3_fa2, V3_o, V3_u
        # b3_i, b3_fp, b3_fa1, b3_fa2, b3_o, b3_u

        self.input_spec = [InputSpec(shape=input_shape)]
        self.input_dim = input_shape[2]
        # initial states: buffer and stack. buffer has shape (samples, buff_limit, output_dim);
        # stack has shape (samples, stack_limit, 2*output_dim)
        self.states = [None, None]

        # The first dims in all weight matrices are k * output_dim because of the recursive nature
        # of treeLSTM
        if self.implementation == 1:
            # Input dimensionality for all W2s is output_dim, and there are 5 W2s: i, fp, fa, u, o
            self.W2 = self.add_weight((self.output_dim, 5 * self.output_dim),
                                      name='{}_W2'.format(self.name), initializer=self.initializer)
            # Input dimensionality for all U2s is output_dim, and there are 5 U2s: i, fp, fa, u, o
            self.U2 = self.add_weight((self.output_dim, 5 * self.output_dim),
                                      name='{}_U2'.format(self.name), initializer=self.initializer)

            # Input dimensionality for all W3s is output_dim, and there are 6 W2s: i, fp, fa1, fa2, u, o
            self.W3 = self.add_weight((self.output_dim, 6 * self.output_dim),
                                      name='{}_W3'.format(self.name), initializer=self.initializer)
            # Input dimensionality for all U3s is output_dim, and there are 6 U3s: i, fp, fa1, fa2, u, o
            self.U3 = self.add_weight((self.output_dim, 6 * self.output_dim),
                                      name='{}_U3'.format(self.name), initializer=self.initializer)
            # Input dimensionality for all V3s is output_dim, and there are 6 V3s: i, fp, fa1, fa2, u, o
            self.V3 = self.add_weight((self.output_dim, 6 * self.output_dim),
                                      name='{}_V3'.format(self.name), initializer=self.initializer)

            self.b2 = K.variable(np.hstack((np.zeros(self.output_dim),
                                            K.get_value(self.add_weight(self.output_dim,
                                                                        initializer=self.forget_bias_initializer)),
                                            K.get_value(self.add_weight(self.output_dim,
                                                                        initializer=self.forget_bias_initializer)),
                                            np.zeros(self.output_dim),
                                            np.zeros(self.output_dim))),
                                 name='{}_b2'.format(self.name))
            self.b3 = K.variable(np.hstack((np.zeros(self.output_dim),
                                            K.get_value(self.add_weight(self.output_dim,
                                                                        initializer=self.forget_bias_initializer)),
                                            K.get_value(self.add_weight(self.output_dim,
                                                                        initializer=self.forget_bias_initializer)),
                                            K.get_value(self.add_weight(self.output_dim,
                                                                        initializer=self.forget_bias_initializer)),
                                            np.zeros(self.output_dim),
                                            np.zeros(self.output_dim))),
                                 name='{}_b3'.format(self.name))
            self.trainable_weights = [self.W2, self.U2, self.W3, self.U3, self.V3, self.b2, self.b3]
        else:
            self.W2_i = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W2_i'.format(self.name),
                                        initializer=self.initializer)
            self.U2_i = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U2_i'.format(self.name),
                                        initializer=self.initializer)
            self.W3_i = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W3_i'.format(self.name),
                                        initializer=self.initializer)
            self.U3_i = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U3_i'.format(self.name),
                                        initializer=self.initializer)
            self.V3_i = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_V3_i'.format(self.name),
                                        initializer=self.initializer)
            self.b2_i = K.zeros((self.output_dim,), name='{}_b2_i'.format(self.name))
            self.b3_i = K.zeros((self.output_dim,), name='{}_b3_i'.format(self.name))

            self.W2_fp = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_W2_fp'.format(self.name),
                                         initializer=self.initializer)
            self.U2_fp = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_U2_fp'.format(self.name),
                                         initializer=self.initializer)
            self.W2_fa = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_W2_fa'.format(self.name),
                                         initializer=self.initializer)
            self.U2_fa = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_U2_fa'.format(self.name),
                                         initializer=self.initializer)
            self.W3_fp = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_W3_fp'.format(self.name),
                                         initializer=self.initializer)
            self.U3_fp = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_U3_fp'.format(self.name),
                                         initializer=self.initializer)
            self.V3_fp = self.add_weight((self.output_dim, self.output_dim),
                                         name='{}_V3_fp'.format(self.name),
                                         initializer=self.initializer)
            self.W3_fa1 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_W3_fa1'.format(self.name),
                                          initializer=self.initializer)
            self.U3_fa1 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_U3_fa1'.format(self.name),
                                          initializer=self.initializer)
            self.V3_fa1 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_V3_fa1'.format(self.name),
                                          initializer=self.initializer)
            self.W3_fa2 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_W3_fa2'.format(self.name),
                                          initializer=self.initializer)
            self.U3_fa2 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_U3_fa2'.format(self.name),
                                          initializer=self.initializer)
            self.V3_fa2 = self.add_weight((self.output_dim, self.output_dim),
                                          name='{}_V3_fa2'.format(self.name),
                                          initializer=self.initializer)
            self.b2_fp = self.add_weight((self.output_dim,), name='{}_b2_fp'.format(self.name),
                                         initializer=self.forget_bias_initializer)
            self.b2_fa = self.add_weight((self.output_dim,), name='{}_b2_fa'.format(self.name),
                                         initializer=self.forget_bias_initializer)
            self.b3_fp = self.add_weight((self.output_dim,), name='{}_b3_fp'.format(self.name),
                                         initializer=self.forget_bias_initializer)
            self.b3_fa1 = self.add_weight((self.output_dim,), name='{}_b3_fa1'.format(self.name),
                                          initializer=self.forget_bias_initializer)
            self.b3_fa2 = self.add_weight((self.output_dim,), name='{}_b3_fa2'.format(self.name),
                                          initializer=self.forget_bias_initializer)

            self.W2_u = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W2_u'.format(self.name),
                                        initializer=self.initializer)
            self.U2_u = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U2_u'.format(self.name),
                                        initializer=self.initializer)
            self.W3_u = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W3_u'.format(self.name),
                                        initializer=self.initializer)
            self.U3_u = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U3_u'.format(self.name),
                                        initializer=self.initializer)
            self.V3_u = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_V3_u'.format(self.name),
                                        initializer=self.initializer)
            self.b2_u = K.zeros((self.output_dim,), name='{}_b2_u'.format(self.name))
            self.b3_u = K.zeros((self.output_dim,), name='{}_b3_u'.format(self.name))

            self.W2_o = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W2_o'.format(self.name),
                                        initializer=self.initializer)
            self.U2_o = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U2_o'.format(self.name),
                                        initializer=self.initializer)
            self.W3_o = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_W3_o'.format(self.name),
                                        initializer=self.initializer)
            self.U3_o = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_U3_o'.format(self.name),
                                        initializer=self.initializer)
            self.V3_o = self.add_weight((self.output_dim, self.output_dim),
                                        name='{}_V3_o'.format(self.name),
                                        initializer=self.initializer)
            self.b2_o = K.zeros((self.output_dim,), name='{}_b2_o'.format(self.name))
            self.b3_o = K.zeros((self.output_dim,), name='{}_b3_o'.format(self.name))

            self.W2 = K.concatenate([self.W2_i, self.W2_fp, self.W2_fa, self.W2_u, self.W2_o])
            self.U2 = K.concatenate([self.U2_i, self.U2_fp, self.U2_fa, self.U2_u, self.U2_o])
            self.W3 = K.concatenate([self.W3_i, self.W3_fp, self.W3_fa1, self.W3_fa2, self.W3_u, self.W3_o])
            self.U3 = K.concatenate([self.U3_i, self.U3_fp, self.U3_fa1, self.U3_fa2, self.U3_u, self.U3_o])
            self.V3 = K.concatenate([self.V3_i, self.V3_fp, self.V3_fa1, self.V3_fa2, self.V3_u, self.V3_o])
            self.b2 = K.concatenate([self.b2_i, self.b2_fp, self.b2_fa, self.b2_u, self.b2_o])
            self.b3 = K.concatenate([self.b3_i, self.b3_fp, self.b3_fa1, self.b3_fa2, self.b3_u, self.b3_o])

        self.regularizers = []
        if self.W_regularizers:
            self.W_regularizers[0].set_param(self.W2)
            self.W_regularizers[1].set_param(self.W3)
            self.regularizers.extend(self.W_regularizers)
        if self.U_regularizers:
            self.U_regularizers[0].set_param(self.U2)
            self.U_regularizers[1].set_param(self.U3)
            self.regularizers.extend(self.U_regularizers)
        if self.V_regularizer:
            self.V_regularizer.set_param(self.V3)
            self.regularizers.append(self.V_regularizer)
        if self.b_regularizers:
            self.b_regularizers[0].set_param(self.b2)
            self.b_regularizers[1].set_param(self.b3)
            self.regularizers.extend(self.b_regularizers)

        self.built = True

    def _one_arg_compose(self, pred_arg):
        # pred_arg: Tensors of size (batch_size, 2, dim) where
        # pred_arg[:,0,:] are arg vectors (h,c) of all samples
        # pred_arg[:,1,:] are pred vectors (h,c) of all samples
        pred_h = pred_arg[:, 1, :self.output_dim]
        pred_c = pred_arg[:, 1, self.output_dim:]
        arg_h = pred_arg[:, 0, :self.output_dim]
        arg_c = pred_arg[:, 0, self.output_dim:]
        if self.implementation == 1:
            # To optimize for GPU, we would want to make fewer
            # matrix multiplications, but with bigger matrices.
            # So we compute outputs of all gates simultaneously
            # using the concatenated operators W2, U2 abd b2
            z_all_gates = K.dot(pred_h, self.W2) + K.dot(arg_h, self.U2) + self.b2  # (batch_size, 5*output_dim)

            # Now picking the appropriate parts for each gate.
            # All five zs are of shape (batch_size, output_dim)
            z_i = z_all_gates[:, :self.output_dim]
            z_fp = z_all_gates[:, self.output_dim: 2*self.output_dim]
            z_fa = z_all_gates[:, 2*self.output_dim: 3*self.output_dim]
            z_u = z_all_gates[:, 3*self.output_dim: 4*self.output_dim]
            z_o = z_all_gates[:, 4*self.output_dim: 5*self.output_dim]

        else:
            # We are optimizing for memory. Smaller matrices, and
            # more computations. So we use the non-concatenated
            # operators W2_i, U2_i, ..
            z_i = K.dot(pred_h, self.W2_i) + K.dot(arg_h, self.U2_i) + self.b2_i
            z_fp = K.dot(pred_h, self.W2_fp) + K.dot(arg_h, self.U2_fp) + self.b2_fp
            z_fa = K.dot(pred_h, self.W2_fa) + K.dot(arg_h, self.U2_fa) + self.b2_fa
            z_u = K.dot(pred_h, self.W2_u) + K.dot(arg_h, self.U2_u) + self.b2_u
            z_o = K.dot(pred_h, self.W2_o) + K.dot(arg_h, self.U2_o) + self.b2_o

        # Applying non-linearity to get outputs of each gate
        i = self.inner_activation(z_i)
        fp = self.inner_activation(z_fp)
        fa = self.inner_activation(z_fa)
        u = self.inner_activation(z_u)
        c = (i * u) + (fp * pred_c) + (fa * arg_c)
        o = self.inner_activation(z_o)

        # Calculate the composition output. SPINN does not have a non-linearity in the
        # following computation, but the original LSTM does.
        h = o * self.activation(c)

        # Finally return the composed representation for the stack, adding a time
        # dimension and make number of dimensions same as the input
        # to this function
        return K.expand_dims(K.concatenate([h, c]), 1)

    def _two_arg_compose(self, pred_args):
        # pred_args: Matrix of size (samples, 3, dim) where
        # pred_args[:,0,:] are arg2 vectors (h,c) of all samples
        # pred_args[:,1,:] are arg1 vectors (h,c) of all samples
        # pred_args[:,2,:] are pred vectors (h,c) of all samples

        # This function is analogous to _one_arg_compose, except that it operates on
        # two args instead of one. Accordingly, the operators are W3, U3, V3 and b3
        # instead of W2, U2 and b2
        pred_h = pred_args[:, 2, :self.output_dim]
        pred_c = pred_args[:, 2, self.output_dim:]
        arg1_h = pred_args[:, 1, :self.output_dim]
        arg1_c = pred_args[:, 1, self.output_dim:]
        arg2_h = pred_args[:, 0, :self.output_dim]
        arg2_c = pred_args[:, 0, self.output_dim:]
        if self.implementation == 1:
            z_all_gates = K.dot(pred_h, self.W3) + K.dot(arg1_h, self.U3) + \
                    K.dot(arg2_h, self.V3) + self.b3  # (batch_size, 6*output_dim)

            z_i = z_all_gates[:, :self.output_dim]
            z_fp = z_all_gates[:, self.output_dim: 2*self.output_dim]
            z_fa1 = z_all_gates[:, 2*self.output_dim: 3*self.output_dim]
            z_fa2 = z_all_gates[:, 3*self.output_dim: 4*self.output_dim]
            z_u = z_all_gates[:, 4*self.output_dim: 5*self.output_dim]
            z_o = z_all_gates[:, 5*self.output_dim: 6*self.output_dim]

        else:
            z_i = K.dot(pred_h, self.W3_i) + K.dot(arg1_h, self.U3_i) + \
                    K.dot(arg2_h, self.V3_i) + self.b3_i
            z_fp = K.dot(pred_h, self.W3_fp) + K.dot(arg1_h, self.U3_fp) + \
                    K.dot(arg2_h, self.V3_fp) + self.b3_fp
            z_fa1 = K.dot(pred_h, self.W3_fa1) + K.dot(arg1_h, self.U3_fa1) + \
                    K.dot(arg2_h, self.V3_fa1) + self.b3_fa1
            z_fa2 = K.dot(pred_h, self.W3_fa2) + K.dot(arg1_h, self.U3_fa2) + \
                    K.dot(arg2_h, self.V3_fa2) + self.b3_fa2
            z_u = K.dot(pred_h, self.W3_u) + K.dot(arg1_h, self.U3_u) + \
                    K.dot(arg2_h, self.V3_u) + self.b3_u
            z_o = K.dot(pred_h, self.W3_o) + K.dot(arg1_h, self.U3_o) + \
                    K.dot(arg2_h, self.V3_o) + self.b3_o

        i = self.inner_activation(z_i)
        fp = self.inner_activation(z_fp)
        fa1 = self.inner_activation(z_fa1)
        fa2 = self.inner_activation(z_fa2)
        u = self.inner_activation(z_u)
        c = (i * u) + (fp * pred_c) + (fa1 * arg1_c) + (fa2 * arg2_c)
        o = self.inner_activation(z_o)

        h = o * self.activation(c)

        return K.expand_dims(K.concatenate([h, c]), 1)

    def step(self, inputs, states):
        # This function is called at each timestep. Before calling this, Keras' rnn
        # dimshuffles the input to have time as the leading dimension, and iterates over
        # it So,
        # inputs: (samples, input_dim) = (samples, x_op + <current timestep's buffer input>)
        #
        # We do not need the current timestep's buffer input here, since the buffer
        # state is the one that's relevant. We just want the current op from inputs.

        buff = states[0] # Current state of buffer
        stack = states[1] # Current state of stack

        step_ops = inputs[:, 0] #(samples, 1), current ops for all samples.

        # We need to make tensors from the ops vectors, one to apply to all dimensions
        # of stack, and the other for the buffer.
        # For the stack:
        # Note stack's dimensionality is 2*output_dim because it holds both h and c
        stack_tiled_step_ops = K.permute_dimensions(
                K.tile(step_ops, (self.stack_limit, 2 * self.output_dim, 1)),
                (2, 0, 1))  # (samples, stack_limit, 2*dim)

        # For the buffer:
        buff_tiled_step_ops = K.permute_dimensions(
                K.tile(step_ops, (self.buffer_ops_limit, 2 * self.output_dim, 1)),
                (2, 0, 1))  # (samples, buff_len, 2*dim)

        shifted_stack = K.concatenate([buff[:, :1], stack], axis=1)[:, :self.stack_limit]
        one_reduced_stack = K.concatenate([self._one_arg_compose(stack[:, :2]),
                                           stack[:, 2:],
                                           K.zeros_like(stack)[:, :1]],
                                          axis=1)
        two_reduced_stack = K.concatenate([self._two_arg_compose(stack[:, :3]),
                                           stack[:, 3:],
                                           K.zeros_like(stack)[:, :2]],
                                          axis=1)
        shifted_buff = K.concatenate([buff[:, 1:], K.zeros_like(buff)[:, :1]], axis=1)

        stack = K.switch(K.equal(stack_tiled_step_ops, SHIFT_OP), shifted_stack, stack)
        stack = K.switch(K.equal(stack_tiled_step_ops, REDUCE2_OP), one_reduced_stack, stack)
        stack = K.switch(K.equal(stack_tiled_step_ops, REDUCE3_OP), two_reduced_stack, stack)
        buff = K.switch(K.equal(buff_tiled_step_ops, SHIFT_OP), shifted_buff, buff)

        stack_top_h = stack[:, 0, :self.output_dim] # first half of the top element for all samples

        return stack_top_h, [buff, stack]

    def get_constants(self, inputs, training=None):
        # TODO(pradeep): The function in the LSTM implementation produces dropout multipliers
        # to apply on the input if dropout is applied on the weights W and U. Ignoring
        # dropout for now.
        constants = []
        if 0 < self.dropout_W < 1 or 0 < self.dropout_U < 1 or 0 < self.dropout_V < 1:
            warnings.warn("Weight dropout not implemented yet. Ignoring them.", RuntimeWarning)
        return constants

    def get_config(self):
        # This function is called to get the model configuration while serializing it
        # Essentially has all the arguments in the __init__ method as a dict.
        config = {'stack_limit': self.stack_limit,
                  'buffer_ops_limit': self.buffer_ops_limit,
                  'output_dim': self.output_dim,
                  'initializer': self.initializer,
                  'forget_bias_initializer': self.forget_bias_initializer,
                  'activation': self.activation.__name__,  # pylint: disable=no-member
                  'inner_activation': self.inner_activation.__name__,  # pylint: disable=no-member
                  'W_regularizer': self.W_regularizers[0].get_config() if self.W_regularizers else None,
                  'U_regularizer': self.U_regularizers[0].get_config() if self.U_regularizers else None,
                  'V_regularizer': self.V_regularizer.get_config() if self.V_regularizer else None,
                  'b_regularizer': self.b_regularizers[0].get_config() if self.b_regularizers else None,
                  'dropout_W': self.dropout_W,
                  'dropout_U': self.dropout_U,
                  'dropout_V': self.dropout_V}
        base_config = super(TreeCompositionLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
