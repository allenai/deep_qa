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
        good_input_layer = Input(shape=good_input.shape[1:], dtype='int32')
        bad_input_layer = Input(shape=bad_input.shape[1:], dtype='int32')
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, name='embedding')
        good_embed = embed_layer(good_input_layer)
        bad_embed = embed_layer(bad_input_layer)
        lstm_layer = LSTM(output_dim=embedding_size, name='lstm')
        good_lstm_out = lstm_layer(good_embed)
        bad_lstm_out = lstm_layer(bad_embed)
        scorer_layer = Dense(1, activation='tanh', name='scorer')
        good_score = scorer_layer(good_lstm_out)
        bad_score = scorer_layer(bad_lstm_out)
        score_diff = merge([good_score, bad_score], mode=lambda scores: scores[1] - scores[0], output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two inputs, since Keras has only supervised losses predefined.
        # But Keras expects a "target" (pseudo_val) in a loss. Just ignore it in the function.
        #TODO: Add a margin on the hinge loss
        score_hinge_loss = lambda pseudo_val, diff: K.mean(diff + 0*pseudo_val, axis=-1)
        model = Model(input=[good_input_layer, bad_input_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='rmsprop')
	model.fit([good_input, bad_input], numpy.zeros((good_input.shape[:1]))) # Note: target is ignored by loss. See above.
        self.model = model

    def score(self, ind):
        if self.model is None:
            raise RuntimeError, "Model not trained yet!"
        embed_layer = None
        lstm_layer = None
        scorer_layer = None
        for layer in self.model.layers:
            if layer.name == "lstm":
                lstm_layer = layer
            elif layer.name == "embedding":
                embed_layer = layer
            elif layer.name == "scorer":
                scorer_layer = layer
        if lstm_layer is None or embed_layer is None or scorer_layer is None:
            raise RuntimeError, "Model does not have expected layers"
        input = Input(shape=ind.shape[1:], dtype='int32')
        score = scorer_layer(lstm_layer(embed_layer(input)))
        scoring_model = Model(input=input, output=score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not going to train.
        scoring_model.compile(loss='mse', optimizer='sgd')
        score_val = scoring_model.predict(ind)
        return score_val

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('train_file', type=str)
    argparser.add_argument('test_file', type=str)
    args = argparser.parse_args()
    lines = [x.strip() for x in open(args.train_file).readlines()]
    test_lines = [x.strip() for x in open(args.test_file).readlines()]
    data_indexer = DataIndexer()
    _, good_input = data_indexer.get_indices(lines, separate_propositions=False)
    test_lengths, test_indices = data_indexer.get_indices(test_lines)
    max_length = max(len(good_input[0]), len(test_indices[0]))
    good_input = data_indexer.pad_indices(good_input, max_length=max_length)
    test_indices = data_indexer.pad_indices(test_indices, max_length=max_length)
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
    for prop_length in test_lengths:
        test_scores.append(test_all_prop_scores[t_ind:t_ind+prop_length].sum())
        t_ind += prop_length
    assert len(test_scores) == len(test_lines)
    outfile = open("out.txt", "w")
    for score, line in zip(test_scores, test_lines):
        print >>outfile, score, line
    outfile.close()
