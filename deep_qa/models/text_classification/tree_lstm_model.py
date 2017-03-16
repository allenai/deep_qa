from typing import Any, Dict
from overrides import overrides

from keras.layers import Input, Dense, Dropout, merge
from keras.regularizers import l2

from ...data.instances.logical_form_instance import LogicalFormInstance
from ...layers.encoders.tree_composition_lstm import TreeCompositionLSTM
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class TreeLSTMModel(TextTrainer):
    """
    This is a model very similar to the TrueFalseModel, in that it simply tries to decide whether
    a statement is true or false.  The difference is that this uses a TreeLSTM to encode logical
    forms, instead of a simple sentence encoder.  This requires logical forms as input, instead of
    regular word sequences.

    We haven't worked on this in a while, so I would not be surprised if it doesn't actually run at
    the moment.
    """
    def __init__(self, params: Dict[str, Any]):
        super(TreeLSTMModel, self).__init__(params)
        self.max_transition_length = None

    def _instance_type(self):
        return LogicalFormInstance

    @overrides
    def _build_model(self):
        '''
        train_input: List of two numpy arrays: transitions and initial buffer
        '''

        # Step 1: Initialze the transition inputs.
        # Length of transitions (ops) is an upper limit on the stack and buffer sizes. So we'll use
        # that to initialize the stack and buffer in the LSTM.
        buffer_ops_limit = self.num_sentence_words
        stack_limit = buffer_ops_limit

        # The transitions input has an extra trailing dimension to make the concatenation with the
        # buffer embedding easier.
        transitions_input = Input(shape=(buffer_ops_limit, 1))

        # Step 2: Convert the logical form input to word vectors.
        logical_form_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="sentence_input")
        logical_form_embeddings = self._embed_input(logical_form_input)

        # Step 3: Merge transitions and buffer.
        lstm_input = merge([transitions_input, logical_form_embeddings], mode='concat')

        # Step 4: Pass the sequences of word vectors through TreeLSTM.
        lstm_layer = TreeCompositionLSTM(stack_limit=stack_limit, buffer_ops_limit=buffer_ops_limit,
                                         output_dim=self.embedding_dim['words'],
                                         W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                                         V_regularizer=l2(0.01), b_regularizer=l2(0.01),
                                         name='treelstm')
        lstm_out = lstm_layer(lstm_input)
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        # Step 5: Find p(true | proposition) by passing the encoded proposition through MLP with
        # ReLU followed by softmax.
        projection_layer = Dense(self.embedding_dim['words']/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        # Step 6: Define crossentropy against labels as the loss and compile the model.
        return DeepQaModel(input=[transitions_input, logical_form_input], output=output_probabilities)

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        max_lengths = super(TreeLSTMModel, self)._get_max_lengths()
        max_lengths['transition_length'] = self.max_transition_length
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        super(TreeLSTMModel, self)._set_max_lengths(max_lengths)
        self.max_transition_length = max_lengths['transition_length']

    @overrides
    def _set_max_lengths_from_model(self):
        self.num_sentence_words = self.model.get_input_shape_at(0)[0][1]
        # TODO(matt): set the max transition length.

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(TreeLSTMModel, cls)._get_custom_objects()
        custom_objects['TreeCompositionLSTM'] = TreeCompositionLSTM
        return custom_objects
