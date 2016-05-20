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

    def train(self, S_g_val, S_b_val, embed_size=50, vocab_size=None):
        if vocab_size is None:
            vocab_size = max(S_g_val.max(), S_b_val.max()) + 1
        S_g = Input(shape=S_g_val.shape[1:], dtype='int32')
        S_b = Input(shape=S_b_val.shape[1:], dtype='int32')
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embed_size, mask_zero=True, name='embedding')
        S_g_embed = embed_layer(S_g)
        S_b_embed = embed_layer(S_b)
        lstm = LSTM(output_dim=embed_size, name='lstm')
        S_g_lstm_out = lstm(S_g_embed)
        S_b_lstm_out = lstm(S_b_embed)
        scorer = Dense(1, activation='tanh', name='scorer')
        g_score = scorer(S_g_lstm_out)
        b_score = scorer(S_b_lstm_out)
        score_diff = merge([g_score, b_score], mode=lambda scores: scores[1] - scores[0], output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two inputs.
        # But Keras expects a "target" (arg0) in a loss. Just ignore it in the function.
        #TODO: Add a margin on the hinge loss
        score_hinge_loss = lambda pseudo_val, diff: K.mean(diff + 0*pseudo_val, axis=-1)
        model = Model(input=[S_g, S_b], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='rmsprop')
        model.fit([S_g_val, S_b_val], numpy.zeros((S_g_val.shape[:1])))
        self.model = model

    def score(self, S_val):
        if self.model is None:
            raise RuntimeError, "Model not trained yet!"
        embedding = None
        lstm = None
        scorer = None
        for layer in self.model.layers:
            if layer.name == "lstm":
                lstm = layer
            elif layer.name == "embedding":
                embedding = layer
            elif layer.name == "scorer":
                scorer = layer
        if lstm is None or embedding is None or scorer is None:
            raise RuntimeError, "Model does not have expected layers"
        S = Input(shape=S_val.shape[1:], dtype='int32')
        S_score = scorer(lstm(embedding(S)))
        scoring_model = Model(input=S, output=S_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not going to train.
        scoring_model.compile(loss='mse', optimizer='sgd')
        S_score_val = scoring_model.predict(S_val)
        return S_score_val

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
