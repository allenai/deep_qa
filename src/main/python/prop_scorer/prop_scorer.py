import sys
import argparse

import numpy
import pickle
from keras.layers import Embedding, Input, LSTM, Dense, merge
from keras.models import Model, model_from_yaml
from keras import backend as K

from index_data import DataIndexer

class PropScorer(object):
    def __init__(self):
        self.model = None
        self.data_indexer = DataIndexer()

    def train(self, good_input, bad_input, embedding_size=50, vocab_size=None):
        '''
        good_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices from sentences in training data
        bad_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices from corrupted versions of sentences in training data
        embedding_size: int. Size of word vectors
        vocab_size: int. Input dimensionality of embedding layer. Will be inferred from inputs if not provided.
        '''
        if vocab_size is None:
            vocab_size = len(self.data_indexer.word_index) + 1

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
        model.fit([good_input, bad_input], numpy.zeros((good_input.shape[:1])), validation_split=0.1) # Note: target is ignored by loss. See above.
        self.model = model

        ## Step 6: Define a scoring model taking the necessary parts of the trained model
        test_input = Input(good_input.shape[1:], dtype='int32')
        test_embed = embed_layer(test_input)
        test_lstm_out = lstm_layer(test_embed)
        test_score = scorer_layer(test_lstm_out)
        self.scoring_model = Model(input=test_input, output=test_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not going to train.
        self.scoring_model.compile(loss='mse', optimizer='sgd')

    def save_model(self, model_name_prefix="prop_scorer"):
        # Serializing the scoring model for future use.
        # Do we need to save the original model as well?
        model_config = self.scoring_model.to_yaml()
        model_config_file = open("%s_config.yaml"%(model_name_prefix), "w")
        print >>model_config_file, model_config
        self.scoring_model.save_weights("%s_weights.h5"%model_name_prefix, overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "w")
        pickle.dump(self.data_indexer, data_indexer_file)

    def load_model(self, model_name_prefix="prop_scorer"):
        # Loading serialized model
        model_config_file = "%s_config.yaml"%(model_name_prefix)
        model_config_yaml = open(model_config_file).read()
        self.scoring_model = model_from_yaml(model_config_yaml)
        self.scoring_model.load_weights("%s_weights.h5"%model_name_prefix)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix)
        self.data_indexer = pickle.load(data_indexer_file)
        self.scoring_model.compile(loss='mse', optimizer='sgd')

    def index_data(self, data_lines, for_train=True, max_length=None):
        if for_train:
            # Indexing training data
            train_lengths, good_input = self.data_indexer.get_indices(data_lines, separate_propositions=False, for_train=for_train)
            # Padding to prespecified length
            print >>sys.stderr, "Indexing training data"
            good_input = self.data_indexer.pad_indices(good_input, max_length=max_length)

            # Corrupting train indices to bet "bad" data
            print >>sys.stderr, "Corrupting training data"
            bad_input = self.data_indexer.corrupt_indices(good_input)

            # Make them both int32 arrays so that Keras will view them as indices.
            good_input = numpy.asarray(good_input, dtype='int32')
            bad_input = numpy.asarray(bad_input, dtype='int32')
            return train_lengths, (good_input, bad_input)
        else:
            # Indexing test data
            print >>sys.stderr, "Indexing test data"
            test_lengths, test_indices = self.data_indexer.get_indices(test_lines, for_train=for_train)
            test_indices = self.data_indexer.pad_indices(test_indices, max_length=max_length)
            test_indices = numpy.asarray(test_indices, dtype='int32')
            return test_lengths, (test_indices,)
        
    def score(self, input):
        score_val = self.scoring_model.predict(input)
        return score_val

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str)
    args = argparser.parse_args()
    prop_scorer = PropScorer()

    if not args.train_file:
        # Training file is not given. There must be a serialized model.
        print >>sys.stderr, "Loading scoring model from disk"
        prop_scorer.load_model()
        # input shape of scoring model is (samples, max_length, word_dim)
        max_length = prop_scorer.scoring_model.get_input_shape_at(0)[1] 
    else:
        print >>sys.stderr, "Reading training data"
        lines = [x.strip() for x in open(args.train_file).readlines()]
        max_length = max([len(line) for line in lines])
        _, (good_input, bad_input) = prop_scorer.index_data(lines, for_train=True, max_length=max_length)
        print >>sys.stderr, "Training model"
        prop_scorer.train(good_input, bad_input)
        prop_scorer.save_model()

    if args.test_file is not None:
        test_lines = [x.strip() for x in open(args.test_file).readlines()]
        test_lengths, (test_indices,) = prop_scorer.index_data(test_lines, max_length=max_length, for_train=False)
        print >>sys.stderr, "Scoring test data"
        test_all_prop_scores = prop_scorer.score(test_indices)
        test_scores = []
        t_ind = 0

        # Iterating through all propositions in the sentence and aggregating their scores
        for num_propositions in test_lengths:
            test_scores.append(test_all_prop_scores[t_ind:t_ind+num_propositions].sum())
            t_ind += num_propositions

        # Once aggregated, the number of scores should be the same as test sentences.
        assert len(test_scores) == len(test_lines)

        outfile = open("out.txt", "w")
        for score, line in zip(test_scores, test_lines):
            print >>outfile, score, line
        outfile.close()
