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

    def train(self, good_ind, bad_ind, embedding_size=50, vocab_size=None):
        if vocab_size is None:
            vocab_size = max(good_ind.max(), bad_ind.max()) + 1
        good_ind_layer = Input(shape=good_ind.shape[1:], dtype='int32')
        bad_ind_layer = Input(shape=bad_ind.shape[1:], dtype='int32')
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, name='embedding')
        good_embed = embed_layer(good_ind_layer)
        bad_embed = embed_layer(bad_ind_layer)
        lstm_layer = LSTM(output_dim=embedding_size, name='lstm')
        good_lstm_out = lstm_layer(good_embed)
        bad_lstm_out = lstm_layer(bad_embed)
        scorer_layer = Dense(1, activation='tanh', name='scorer')
        good_score = scorer_layer(good_lstm_out)
        bad_score = scorer_layer(bad_lstm_out)
        score_diff = merge([good_score, bad_score], mode=lambda scores: scores[1] - scores[0], output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two inputs.
        # But Keras expects a "target" (arg0) in a loss. Just ignore it in the function.
        #TODO: Add a margin on the hinge loss
        score_hinge_loss = lambda pseudo_val, diff: K.mean(diff + 0*pseudo_val, axis=-1)
        model = Model(input=[good_ind_layer, bad_ind_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='rmsprop')
        model.fit([good_ind, bad_ind], numpy.zeros((good_ind.shape[:1])))
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
    di = DataIndexer()
    _, inds = di.get_indices(lines, separate_props=False)
    test_prop_lens, test_inds = di.get_indices(test_lines)
    maxlen = max(len(inds[0]), len(test_inds[0]))
    inds = di.pad_indices(inds, maxlen=maxlen)
    test_inds = di.pad_indices(test_inds, maxlen=maxlen)
    corrupt_inds = di.corrupt_indices(inds)
    g_inds = numpy.asarray(inds, dtype='int32')
    b_inds = numpy.asarray(corrupt_inds, dtype='int32')
    test_inds = numpy.asarray(test_inds, dtype='int32')
    vocab_size = max(g_inds.max(), b_inds.max(), test_inds.max()) + 1
    ps = PropScorer()
    ps.train(g_inds, b_inds, vocab_size=vocab_size)
    test_all_prop_scores = ps.score(test_inds)
    test_scores = []
    t_ind = 0
    for prop_len in test_prop_lens:
        test_scores.append(test_all_prop_scores[t_ind:t_ind+prop_len].sum())
        t_ind += prop_len
    assert len(test_scores) == len(test_lines)
    outfile = open("out.txt", "w")
    for score, line in zip(test_scores, test_lines):
        print >>outfile, score, line
    outfile.close()
