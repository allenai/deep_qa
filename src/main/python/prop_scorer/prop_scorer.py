import sys
import argparse

import numpy
from keras.layers import Embedding, Input, LSTM, Dense, merge
from keras.models import Model
from keras import backend as K

from index_data import DataIndexer

class PropScorer(object):
    def __init__(self):
        self.model = None

    def train(self, good_input, bad_input, embedding_size=50, vocab_size=None):
        '''
        good_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices from sentences in training data
        bad_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices from corrupted versions of sentences in training data
        embedding_size: int. Size of word vectors
        vocab_size: int. Input dimensionality of embedding layer. Will be inferred from inputs if not provided.
        '''
        if vocab_size is None:
            vocab_size = max(good_input.max(), bad_input.max()) + 1

        ## STEP 1: Initialze the two inputs
        good_input_layer = Input(shape=good_input.shape[1:], dtype='int32')
        bad_input_layer = Input(shape=bad_input.shape[1:], dtype='int32')

        ## STEP 2: Embed the two propositions by changing word indices to word vectors
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, name='embedding')
        # Share embedding layer for both propositions
        good_embed = embed_layer(good_input_layer)
        bad_embed = embed_layer(bad_input_layer)

        ## STEP 3: Pass the sequences of word vectors through the same LSTM
        lstm_layer = LSTM(output_dim=embedding_size, name='lstm')
        # Share LSTM layer for both propositions
        good_lstm_out = lstm_layer(good_embed)
        bad_lstm_out = lstm_layer(bad_embed)

        ## STEP 4: Score the two propositions by passing the outputs from LSTM twough the same dense layer
        #TODO: Can make the scorer more complex by adding more layers
        scorer_layer = Dense(1, activation='tanh', name='scorer')
        # Share scoring layer for both propositions
        good_score = scorer_layer(good_lstm_out)
        bad_score = scorer_layer(bad_lstm_out)

        ## Step 5: Define the score difference as the loss and train the full model jointly
        score_diff = merge([good_score, bad_score], mode=lambda scores: scores[1] - scores[0], output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two inputs, since Keras has only supervised losses predefined.
        # But Keras expects a "target" (dummy_target) in a loss. Just ignore it in the function. Technically dummy_target is an argument of the final function, and Theano will complain if it is not a part of the computational graph. So, resorting to this hack of 0*dummy_target
        #TODO: Add a margin on the hinge loss
        score_hinge_loss = lambda dummy_target, diff: K.mean(diff + 0*dummy_target, axis=-1)
        model = Model(input=[good_input_layer, bad_input_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='rmsprop')
        model.fit([good_input, bad_input], numpy.zeros((good_input.shape[:1]))) # Note: target is ignored by loss. See above.
        self.model = model
        test_input = Input(good_input.shape[1:], dtype='int32')
        test_embed = embed_layer(test_input)
        test_lstm_out = lstm_layer(test_embed)
        test_score = scorer_layer(test_lstm_out)
        self.scoring_model = Model(input=test_input, output=test_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not going to train.
        self.scoring_model.compile(loss='mse', optimizer='sgd')

    def score(self, input):
        score_val = self.scoring_model.predict(input)
        return score_val

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('train_file', type=str)
    argparser.add_argument('test_file', type=str)
    args = argparser.parse_args()
    lines = [x.strip() for x in open(args.train_file).readlines()]
    test_lines = [x.strip() for x in open(args.test_file).readlines()]
    data_indexer = DataIndexer()
    
    # Indexing training data
    _, good_input = data_indexer.get_indices(lines, separate_propositions=False)
    
    # Indexing test data
    test_lengths, test_indices = data_indexer.get_indices(test_lines)
    
    # Pad both the indices to make sure that the time dimension is of the same length
    max_length = max(len(good_input[0]), len(test_indices[0]))
    good_input = data_indexer.pad_indices(good_input, max_length=max_length)
    test_indices = data_indexer.pad_indices(test_indices, max_length=max_length)
    
    # Corrupting train indices to bet "bad" data
    bad_input = data_indexer.corrupt_indices(good_input)

    good_input = numpy.asarray(good_input, dtype='int32')
    bad_input = numpy.asarray(bad_input, dtype='int32')
    test_indices = numpy.asarray(test_indices, dtype='int32')
    vocab_size = max(good_input.max(), bad_input.max(), test_indices.max()) + 1
    prop_scorer = PropScorer()
    prop_scorer.train(good_input, bad_input, vocab_size=vocab_size)
    test_all_prop_scores = prop_scorer.score(test_indices)
    test_scores = []
    t_ind = 0
    for num_propositions in test_lengths:
        test_scores.append(test_all_prop_scores[t_ind:t_ind+num_propositions].sum())
        t_ind += num_propositions
    assert len(test_scores) == len(test_lines)
    outfile = open("out.txt", "w")
    for score, line in zip(test_scores, test_lines):
        print >>outfile, score, line
    outfile.close()
