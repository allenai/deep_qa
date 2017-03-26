import warnings

from keras import backend as K
from keras import activations
from keras.engine import InputSpec
from keras.layers import LSTM
from overrides import overrides

class KnowledgeBackedLSTM(LSTM):
    '''
    A KnowledgeBackedLSTM is a variant of an LSTM that takes additional background information as a
    matrix along with the input vector at each timestep, computes a weighted average of the matrix
    to get another vector, and feeds it along with the actual input vector to the LSTM. The weights
    used for the average are computed as a function of the previous timestep's output, and the
    background information matrix.
    '''
    def __init__(self, units, token_dim, knowledge_dim, knowledge_length,
                 attention_init='uniform', attention_activation='tanh', **kwargs):
        """
        output_dim (int): Dimensionality of output (same as LSTM)
        token_dim (int): Input dimensionality of token embeddings
        knowledge_dim (int): Input dimensionality of background info
        knowledge_length (int): Number of units of background information
            provided per token
        attention_init (str): Initialization heuristic for attention scorer
        attention_activation (str): Activation used at hidden layer in the
            attention MLP
        """
        self.token_dim = token_dim
        self.knowledge_dim = knowledge_dim
        self.knowledge_length = knowledge_length
        self.attention_init = attention_init
        self.attention_activation = activations.get(attention_activation)
        # LSTM's constructor expects output_dim. So pass it along.
        kwargs['units'] = units
        super(KnowledgeBackedLSTM, self).__init__(**kwargs)
        # This class' grand parent (Recurrent) would have set ndim (number of
        # input dimensions) to 3. Let's change that to 4.
        self.input_spec = [InputSpec(ndim=4)]
        if self.implementation == 0:
            # Keras' implementation of LSTM precomputes the inputs to all gates
            # to save CPU. However, in this implementation, part of the input is
            # a weighted average of the background knowledge, with the weights being
            # a function of the output of the previous time step. So the
            # precomputation cannot be done, making consume_less = cpu meaningless.
            warnings.warn("Current implementation does not support consume_less=cpu. \
                    Ignoring the setting.")
            self.consume_less = "mem"

    @overrides
    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        # projection_dim is the size of the hidden layer in the MLP
        # used for attention
        projection_dim = (self.token_dim + self.knowledge_dim) / 2
        self.token_projector = self.add_weight((self.token_dim, projection_dim),
                                               name='{}_token_projector'.format(self.name),
                                               initializer=self.kernel_initializer)
        self.knowledge_projector = self.add_weight((self.knowledge_dim, projection_dim),
                                                   name='{}_knowledge_projector'.format(self.name),
                                                   initializer=self.kernel_initializer)
        self.attention_scorer = self.add_weight((projection_dim,),
                                                name='{}_attention_scorer'.format(self.name),
                                                initializer=self.attention_init)
        # Initialize the LSTM parameters by passing the appropriate
        # shape to its build method.
        # A weighted average of the knowledge matrix will be concatenated to the
        # input vector at each timestep and passed to the LSTM as input.
        # So the LSTM's input_dim = token_dim + knowledge_dim
        lstm_input_shape = input_shape[:2] + (self.token_dim + self.knowledge_dim,)
        super(KnowledgeBackedLSTM, self).build(lstm_input_shape)
        # LSTM's build method sets the shape attribute of InputSpec, thus
        # changing the ndim (again!). Let's reset it.
        self.input_spec = [InputSpec(shape=input_shape)]

        self.built = True

    @overrides
    def step(self, inputs, states):
        # While the actual input to the layer is of
        # shape (batch_size, time, knowledge_length, token_dim+knowledge_dim), inputs in this function
        # is of shape (batch_size, knowledge_length, token_dim+knowledge_dim) as Keras iterates over
        # the time dimension and calls this function once per timestep.

        # TODO(matt): this variable is not used.  Should we be using the previous hidden state in
        # here?
        h_tm1 = states[0]  # Output from previous (t-1) timestep; pylint: disable=unused-variable
        token_t = inputs[:, 0, :self.token_dim]  # Current token (batch_size, token_dim)

        # Repeated along knowledge_len (batch_size, knowledge_len, token_dim)
        tiled_token_t = inputs[:, :, :self.token_dim]
        knowledge_t = inputs[:, :, self.token_dim:]  # Current knowledge (batch_size, knowledge_len, knowledge_dim)

        # TODO(pradeep): Try out other kinds of interactions between knowledge and tokens.
        # Candidates: dot product, difference, element wise product, inner product ..
        projected_combination = self.attention_activation(
                K.dot(knowledge_t, self.knowledge_projector) +
                K.dot(tiled_token_t, self.token_projector)) # (batch_size, knowledge_len, proj_dim)
        # Shape: (batch_size, knowledge_len)
        attention_scores = K.softmax(K.dot(projected_combination, self.attention_scorer))

        # Add a dimension at the end for attention scores to make the number of
        # dimensions the same as that of knowledge_t, multiply and compute sum along knowledge_len to
        # get a weighted average of all pieces of background information.
        # Shape: (batch_size, knowledge_dim)
        attended_knowledge = K.sum(knowledge_t * K.expand_dims(attention_scores, -1), axis=1)
        lstm_input_t = K.concatenate([token_t, attended_knowledge])  # (batch_size, tok_dim+knowledge_dim)

        # Now pass the concatenated input to LSTM's step function like nothing ever happened.
        return super(KnowledgeBackedLSTM, self).step(lstm_input_t, states)

    @overrides
    def get_constants(self, inputs, training=None):
        # overriding this function from LSTM because we have an extra dimension knowledge_len which
        # needs to be ignored while initializing h and c.  This function computes weight dropouts.
        # Since these are specific to LSTM, we don't have to worry about the rest of the input.
        lstm_input = inputs[:, :, 0, :]
        return super(KnowledgeBackedLSTM, self).get_constants(lstm_input)

    @overrides
    def get_initial_states(self, inputs):
        # overriding this method from Recurrent because we have an extra dimension
        # knowledge_len which needs to be ignored while initializing h and c.
        lstm_input = inputs[:, :, 0, :]
        return super(KnowledgeBackedLSTM, self).get_initial_states(lstm_input)

    @overrides
    def get_config(self):
        config = {
                'token_dim': self.token_dim,
                'knowledge_dim': self.knowledge_dim,
                'knowledge_length': self.knowledge_length,
                'attention_init': self.attention_init,
                'attention_activation': self.attention_activation.__name__,  # pylint: disable=no-member
                }
        base_config = super(KnowledgeBackedLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
