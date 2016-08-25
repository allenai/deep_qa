import sys

import numpy
from keras.layers import Input, Dense, Dropout, merge
from keras.models import Model
from keras.regularizers import l2

from .nn_solver import NNSolver
from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from ..layers.encoders import TreeCompositionLSTM

class TreeLSTMSolver(NNSolver):
    def __init__(self, **kwargs):
        super(TreeLSTMSolver, self).__init__(**kwargs)

    def _build_model(self, train_input):
        '''
        train_input: List of two numpy arrays: transitions and initial buffer
        '''
        transitions, logical_forms = train_input

        # Step 1: Initialze the transition inputs.
        # Length of transitions (ops) is an upper limit on the stack and buffer sizes. So we'll use
        # that to initialize the stack and buffer in the LSTM.
        buffer_ops_limit = transitions.shape[1]
        stack_limit = buffer_ops_limit

        # The transitions input has an extra trailing dimension to make the concatenation with the
        # buffer embedding easier.
        transitions_input = Input(shape=(buffer_ops_limit, 1))

        # Step 2: Convert the logical form input to word vectors.
        logical_form_input_layer, logical_form_embeddings = self._get_embedded_sentence_input(
                logical_forms)

        # Step 3: Merge transitions and buffer.
        lstm_input = merge([transitions_input, logical_form_embeddings], mode='concat')

        # Step 4: Pass the sequences of word vectors through TreeLSTM.
        lstm_layer = TreeCompositionLSTM(stack_limit, buffer_ops_limit,
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
        print(model.summary(), file=sys.stderr)
        return model

    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]

    def prep_labeled_data(self, dataset: TextDataset, for_train: bool):
        processed_dataset = self._index_and_pad_dataset(dataset, [self.max_sentence_length])
        sentence_inputs, labels = processed_dataset.as_training_data()

        # TODO(matt): we need to pad the transitions / element sequences to the same length.  As it
        # is, this code does not work.  Probably the right thing to do is to have a new kind of
        # Instance, a TreeInstance, or something, and it computes the shift/reduce sequences, and
        # does the padding correctly.  Then as_training_data() on that dataset gives the
        # transitions, elements, and labels, and none of this has to be done here.
        transitions, elements = self.data_indexer.get_shift_reduce_sequences(sentence_inputs)
        if for_train:
            self.max_sentence_length = max(len(t) for t in transitions)

        # Make int32 array so that Keras will view them as indices.
        elements = numpy.asarray(elements, dtype='int32')

        # TreeLSTM's transitions input has an extra trailing dimension for
        # concatenation. This has to match.
        transitions = numpy.expand_dims(transitions, axis=-1)

        return (transitions, elements), numpy.asarray(labels)

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(TreeLSTMSolver, cls)._get_custom_objects()
        custom_objects['TreeCompositionLSTM'] = TreeCompositionLSTM
        return custom_objects
