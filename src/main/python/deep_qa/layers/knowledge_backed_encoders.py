'''
Knowledge backed models are variants of LSTM cells that take additional background
information as a matrix along with the input vector at each timestep, compute a weighted
average of the matrix to get another vector, and feed it along with the actual input
vector to the LSTM. The weights used for the average are computed as a function of the
previous timestep's output, and the background information matrix.
'''

import warnings

from keras import backend as K
from keras import initializations, activations
from keras.engine import InputSpec
from keras.layers import LSTM
from .encoders import TreeCompositionLSTM

class KnowledgeBackedLSTM(LSTM):
    def __init__(self, output_dim, token_dim, knowledge_dim, knowledge_length,
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
        self.attention_init = initializations.get(attention_init)
        self.attention_activation = activations.get(attention_activation)
        # LSTM's constructor expects output_dim. So pass it along.
        kwargs['output_dim'] = output_dim
        super(KnowledgeBackedLSTM, self).__init__(**kwargs)
        # This class' grand parent (Recurrent) would have set ndim (number of
        # input dimensions) to 3. Let's change that to 4.
        self.input_spec = [InputSpec(ndim=4)]
        if self.consume_less == 'cpu':
            # Keras' implementation of LSTM precomputes the inputs to all gates
            # to save CPU. However, in this implementation, part of the input is
            # a weighted average of the background knowledge, with the weights being
            # a function of the output of the previous time step. So the
            # precomputation cannot be done, making consume_less = cpu meaningless.
            warnings.warn("Current implementation does not support consume_less=cpu. \
                    Ignoring the setting.")
            self.consume_less = "mem"

    def build(self, input_shape):
        # pylint: disable=attribute-defined-outside-init
        # projection_dim is the size of the hidden layer in the MLP
        # used for attention
        projection_dim = (self.token_dim + self.knowledge_dim) / 2
        self.token_projector = self.init((self.token_dim, projection_dim),
                                         name='{}_token_projector'.format(self.name))
        self.knowledge_projector = self.init((self.knowledge_dim, projection_dim),
                                             name='{}_knowledge_projector'.format(self.name))
        self.attention_scorer = self.attention_init((projection_dim,),
                                                    name='{}_attention_scorer'.format(self.name))
        # If any initial weights were passed to the constructor, we have to set
        # them here. LSTM's build method needs to be called before that to have
        # all the trainable weight variables initialzed. However, LSTM's build
        # method sees that the inital weights array is not None, tries to set it
        # and then delete the array. To avoid this, let's make a copy of the
        # weights, deal with them later, and make the original array None.
        self.initial_kblstm_weights = self.initial_weights
        self.initial_weights = None

        # Initialize the LSTM parameters by passing the appropriate
        # shape to its build method.
        # A weighted average of the knowledge matrix will be concatenated to the
        # input vector at each timestep and passed to the LSTM as input.
        # So the LSTM's input_dim = token_dim + knowledge_dim
        lstm_input_shape = input_shape[:2] + (self.token_dim+self.knowledge_dim,)
        super(KnowledgeBackedLSTM, self).build(lstm_input_shape)
        # LSTM's build method sets the shape attribute of InputSpec, thus
        # changing the ndim (again!). Let's reset it.
        self.input_spec = [InputSpec(shape=input_shape)]

        # self.trainable_weights array will now have LSTM's weights. Let's add
        # to it.
        self.trainable_weights.extend([self.token_projector,
                                       self.knowledge_projector,
                                       self.attention_scorer])
        # Now that trainable_weights is complete, let's set weights if needed.
        if self.initial_kblstm_weights is not None:
            self.set_weights(self.initial_kblstm_weights)
            del self.initial_kblstm_weights

    def step(self, x, states):
        # While the actual input to the layer is of
        # shape (batch_size, time, knowledge_length, token_dim+knowledge_dim), x in this function
        # is of shape (batch_size, knowledge_length, token_dim+knowledge_dim) as Keras iterates over
        # the time dimension and calls this function once per timestep.

        # TODO(matt): this variable is not used.  Should we be using the previous hidden state in
        # here?
        h_tm1 = states[0]  # Output from previous (t-1) timestep
        token_t = x[:, 0, :self.token_dim]  # Current token (batch_size, token_dim)

        # Repeated along knowledge_len (batch_size, knowledge_len, token_dim)
        tiled_token_t = x[:, :, :self.token_dim]
        knowledge_t = x[:, :, self.token_dim:]  # Current knowledge (batch_size, knowledge_len, knowledge_dim)

        # TODO(pradeep): Try out other kinds of interactions between knowledge and tokens.
        # Candidates: dot product, difference, element wise product, inner product ..
        projected_combination = self.attention_activation(
                K.dot(knowledge_t, self.knowledge_projector) +
                K.dot(tiled_token_t, self.token_projector)) # (batch_size, knowledge_len, proj_dim)
        attention_scores = K.softmax(K.dot(projected_combination, self.attention_scorer))  # (batch_size, knowledge_len)

        # Add a dimension at the end for attention scores to make the number of
        # dimensions the same as that of knowledge_t, multiply and compute sum along knowledge_len to
        # get a weighted average of all pieces of background information.
        attended_knowledge = K.sum(knowledge_t * K.expand_dims(attention_scores, -1), axis=1)  # (batch_size, knowledge_dim)
        lstm_input_t = K.concatenate([token_t, attended_knowledge])  # (batch_size, tok_dim+knowledge_dim)

        # Now pass the concatenated input to LSTM's step function like nothing ever happened.
        return super(KnowledgeBackedLSTM, self).step(lstm_input_t, states)

    def get_constants(self, x):
        # overriding this function from LSTM because we have an extra dimension knowledge_len which
        # needs to be ignored while initializing h and c.  This function computes weight dropouts.
        # Since these are specific to LSTM, we don't have to worry about the rest of the input.
        lstm_input = x[:, :, 0, :]
        return super(KnowledgeBackedLSTM, self).get_constants(lstm_input)

    def get_initial_states(self, x):
        # overriding this method from Recurrent because we have an extra dimension
        # knowledge_len which needs to be ignored while initializing h and c.
        lstm_input = x[:, :, 0, :]
        return super(KnowledgeBackedLSTM, self).get_initial_states(lstm_input)

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'token_dim': self.token_dim,
                  'knowledge_dim': self.knowledge_dim,
                  'knowledge_length': self.knowledge_length,
                  'attention_init':self.attention_init.__name__,
                  'attention_activation':self.attention_activation.__name__}
        base_config = super(KnowledgeBackedLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class KnowledgeBackedTreeLSTM(TreeCompositionLSTM):
    # TODO(pradeep): Implement me.
    def __init__(self):  # pylint: disable=super-init-not-called
        raise NotImplementedError
