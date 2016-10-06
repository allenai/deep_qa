from typing import Any, Dict
from overrides import overrides

from keras.layers import Input, Dense, Dropout, merge
from keras.models import Model
from keras.regularizers import l2

from .nn_solver import NNSolver
from ..data.text_instance import LogicalFormInstance
from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from ..layers.encoders import TreeCompositionLSTM


class TreeLSTMSolver(NNSolver):
    def __init__(self, params: Dict[str, Any]):
        super(TreeLSTMSolver, self).__init__(params)
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
        buffer_ops_limit = self.max_sentence_length
        stack_limit = buffer_ops_limit

        # The transitions input has an extra trailing dimension to make the concatenation with the
        # buffer embedding easier.
        transitions_input = Input(shape=(buffer_ops_limit, 1))

        # Step 2: Convert the logical form input to word vectors.
        logical_form_input_layer, logical_form_embeddings = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,), name_prefix="sentence")

        # Step 3: Merge transitions and buffer.
        lstm_input = merge([transitions_input, logical_form_embeddings], mode='concat')

        # Step 4: Pass the sequences of word vectors through TreeLSTM.
        lstm_layer = TreeCompositionLSTM(stack_limit=stack_limit, buffer_ops_limit=buffer_ops_limit,
                                         output_dim=self.embedding_size, W_regularizer=l2(0.01),
                                         U_regularizer=l2(0.01), V_regularizer=l2(0.01),
                                         b_regularizer=l2(0.01), name='treelstm')
        lstm_out = lstm_layer(lstm_input)
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        # Step 5: Find p(true | proposition) by passing the encoded proposition through MLP with
        # ReLU followed by softmax.
        projection_layer = Dense(self.embedding_size/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        # Step 6: Define crossentropy against labels as the loss and compile the model.
        model = Model(input=[transitions_input, logical_form_input_layer], output=output_probabilities)
        return model

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'transition_length': self.max_transition_length,
                }

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_transition_length = max_lengths['transition_length']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]
        # TODO(matt): set the max transition length.

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(TreeLSTMSolver, cls)._get_custom_objects()
        custom_objects['TreeCompositionLSTM'] = TreeCompositionLSTM
        return custom_objects
