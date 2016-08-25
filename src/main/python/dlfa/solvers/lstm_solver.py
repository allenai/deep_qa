import sys

import numpy

from keras.layers import LSTM, Dense, Dropout
from keras.models import Model
from keras.regularizers import l2

from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from .nn_solver import NNSolver

class LSTMSolver(NNSolver):
    """
    An NNSolver that simply takes word sequences as input (could be either sentences or logical
    forms), encodes the sequence using an LSTM, then uses a few dense layers to decide if the
    sentence encoding is true or false.

    We don't really expect this model to work.  The best it can do is basically to learn word
    cooccurrence information, similar to how the Salience solver works, and I'm not at all
    confident that this does that job better than Salience.  We've implemented this mostly as a
    simple baseline.
    """
    def __init__(self, **kwargs):
        super(LSTMSolver, self).__init__(**kwargs)

    def _build_model(self, train_input):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        # Step 1: Convert the sentence input into sequences of word vectors.
        input_layer, sentence_embedding = self._get_embedded_sentence_input(train_input)

        # Step 2: Pass the sequences of word vectors through LSTM.
        lstm_layer = LSTM(output_dim=self.embedding_size, W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                          b_regularizer=l2(0.01), name='lstm')
        lstm_out = lstm_layer(sentence_embedding)

        # Add a dropout after LSTM.
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        # Step 3: Find p(true | proposition) by passing the outputs from LSTM through an MLP with
        # ReLU layers.
        projection_layer = Dense(int(self.embedding_size/2), activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        # Step 4: Define crossentropy against labels as the loss.
        model = Model(input=input_layer, output=output_probabilities)
        print(model.summary(), file=sys.stderr)
        return model

    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]

    def prep_labeled_data(self, dataset: TextDataset, for_train: bool):
        processed_dataset = self._index_and_pad_dataset(dataset, [self.max_sentence_length])
        if for_train:
            self.max_sentence_length = processed_dataset.max_lengths()[0]
        inputs, labels = processed_dataset.as_training_data()
        return numpy.asarray(inputs), numpy.asarray(labels)
