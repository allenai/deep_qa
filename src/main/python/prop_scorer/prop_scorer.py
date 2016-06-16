import sys
import argparse
import random
import numpy
import codecs
import pickle
from keras.layers import Embedding, Input, LSTM, Dense, Dropout, merge
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras import backend as K

from index_data import DataIndexer

class PropScorer(object):
    def __init__(self):
        self.model = None
        self.data_indexer = DataIndexer()

    def train(self, good_input, bad_input, embedding_size=50, vocab_size=None):
        '''
        good_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from sentences in training data
        bad_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from corrupted versions of sentences in training data
        embedding_size: int. Size of word vectors
        vocab_size: int. Input dimensionality of embedding layer. Will be inferred from inputs 
            if not provided.
        '''
        if vocab_size is None:
            vocab_size = self.data_indexer.get_vocab_size()

        ## STEP 1: Initialze the two inputs
        good_input_layer = Input(shape=good_input.shape[1:], dtype='int32')
        bad_input_layer = Input(shape=bad_input.shape[1:], dtype='int32')

        ## STEP 2: Embed the two propositions by changing word indices to word vectors
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, mask_zero=True, 
                name='embedding')
        # Share embedding layer for both propositions
        good_embed = embed_layer(good_input_layer)
        bad_embed = embed_layer(bad_input_layer)
        # Add a dropout to regularize the input representations
        regularized_good_embed = Dropout(0.5)(good_embed)
        regularized_bad_embed = Dropout(0.5)(bad_embed)

        ## STEP 3: Pass the sequences of word vectors through the same LSTM
        lstm_layer = LSTM(output_dim=embedding_size, W_regularizer=l2(0.01), U_regularizer=l2(0.01), b_regularizer=l2(0.01), name='lstm')
        # Share LSTM layer for both propositions
        good_lstm_out = lstm_layer(regularized_good_embed)
        bad_lstm_out = lstm_layer(regularized_bad_embed)
	regularized_good_lstm_out = Dropout(0.2)(good_lstm_out)
	regularized_bad_lstm_out = Dropout(0.2)(bad_lstm_out)

        ## STEP 4: Score the two propositions by passing the outputs from LSTM twough the same 
        # dense layer
        #TODO: Can make the scorer more complex by adding more layers
        scorer_layer = Dense(1, activation='tanh', W_regularizer=l2(0.01), b_regularizer=l2(0.01), name='scorer')
        # Share scoring layer for both propositions
        good_score = scorer_layer(regularized_good_lstm_out)
        bad_score = scorer_layer(regularized_bad_lstm_out)

        ## Step 5: Define the score difference as the loss and compile the model
        score_diff = merge([good_score, bad_score], mode=lambda scores: scores[1] - scores[0], 
                output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two 
        # inputs, since Keras has only supervised losses predefined.
        # But Keras expects a "target" (dummy_target) in a loss. Just ignore it in the function. 
        #Technically dummy_target is an argument of the final function, and Theano will complain 
        # if it is not a part of the computational graph. So, resorting to this hack of 0*dummy_target
        #TODO: Add a margin on the hinge loss
        score_hinge_loss = lambda dummy_target, diff: K.mean(diff + 0*dummy_target, axis=-1)
        model = Model(input=[good_input_layer, bad_input_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='adam')
        print >>sys.stderr, model.summary()

        ## Step 6: Train the full model jointly
        # Define early stopping with patience of 1 (will wait for atmost one epoch after validation loss 
        # stops decreasing). The quantity being monitored is the validation loss (default option).
        early_stopping = EarlyStopping(patience=1)
        # Separate the last 10% of training data as validation data, which is used for early stopping
        model.fit([good_input, bad_input], numpy.zeros((good_input.shape[:1])), validation_split=0.1, 
                callbacks=[early_stopping]) # Note: target is ignored by loss. See above.
        self.model = model

        ## Step 6: Define a scoring model taking the necessary parts of the trained model
        test_input = Input(good_input.shape[1:], dtype='int32')
        test_embed = embed_layer(test_input)
        test_lstm_out = lstm_layer(test_embed)
        test_score = scorer_layer(test_lstm_out)
        self.scoring_model = Model(input=test_input, output=test_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not 
        # going to train.
        self.scoring_model.compile(loss='mse', optimizer='sgd')

    def save_model(self, model_name_prefix):
        # Serializing the scoring model for future use.
        model_config = self.scoring_model.to_json()
        model_config_file = open("%s_config.json"%(model_name_prefix), "w")
        print >>model_config_file, model_config
        self.scoring_model.save_weights("%s_weights.h5"%model_name_prefix, overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        model_config_file.close()
        data_indexer_file.close()

    def load_model(self, model_name_prefix):
        # Loading serialized model
        model_config_file = open("%s_config.json"%(model_name_prefix))
        model_config_json = model_config_file.read()
        self.scoring_model = model_from_json(model_config_json)
        self.scoring_model.load_weights("%s_weights.h5"%model_name_prefix)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.scoring_model.compile(loss='mse', optimizer='sgd')
        model_config_file.close()
        data_indexer_file.close()

    def prepare_training_data(self, data_lines, max_length=None):
        # max_length is used for ignoring long training instances
        # and also padding all instances to make them of the same 
        # length

        # Indexing training data
        print >>sys.stderr, "Indexing training data"
        num_train_propositions, good_input = self.data_indexer.process_data(data_lines, 
                separate_propositions=False, for_train=True, max_length=max_length)
        # Padding to prespecified length
        good_input = self.data_indexer.pad_indices(good_input, max_length=max_length)

        # Corrupting train indices to get "bad" data
        print >>sys.stderr, "Corrupting training data"
        bad_input = self.data_indexer.corrupt_indices(good_input)

        # Print a training sample
        print >>sys.stderr, "Sample training pairs:"
        for good_prop_indices, bad_prop_indices in zip(good_input, bad_input)[:10]:
            print >>sys.stderr, "%s vs. %s"%(self.data_indexer.get_words_from_indices(good_prop_indices), 
                    self.data_indexer.get_words_from_indices(bad_prop_indices))

        # Make int32 array so that Keras will view them as indices.
        good_input = numpy.asarray(good_input, dtype='int32')
        bad_input = numpy.asarray(bad_input, dtype='int32')
        return num_train_propositions, (good_input, bad_input)

    def prepare_test_data(self, data_lines, max_length=None):
        # Indexing test data
        print >>sys.stderr, "Indexing test data"
        num_test_propositions, test_indices = self.data_indexer.process_data(test_lines, 
                for_train=False)
        test_indices = self.data_indexer.pad_indices(test_indices, max_length=max_length)
        # Make int32 array so that Keras will view them as indices.
        test_indices = numpy.asarray(test_indices, dtype='int32')
        return num_test_propositions, test_indices
        
    def score(self, input):
        score_val = self.scoring_model.predict(input)
        return score_val

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--length_upper_limit', type=int, 
            help="Upper limit on length of training data. Ignored during testing.")
    argparser.add_argument('--max_train_size', type=int, 
            help="Upper limit on the size of training data")
    args = argparser.parse_args()
    prop_scorer = PropScorer()

    if not args.train_file:
        # Training file is not given. There must be a serialized model.
        print >>sys.stderr, "Loading scoring model from disk"
        prop_scorer.load_model("prop_scorer")
        # input shape of scoring model is (samples, max_length)
    else:
        print >>sys.stderr, "Reading training data"
        lines = [x.strip() for x in open(args.train_file).readlines()]
        # Shuffling lines now since Keras does not shuffle data before validation
        # split is made. This will ensure validation data is not at the end of the file.
        random.shuffle(lines)
        _, (good_input, bad_input) = prop_scorer.prepare_training_data(lines, 
                max_length=args.length_upper_limit)
        if args.max_train_size is not None:
            print >>sys.stderr, "Limiting training size to %d"%(args.max_train_size)
            good_input = good_input[:args.max_train_size]
            bad_input = bad_input[:args.max_train_size]
        print >>sys.stderr, "Training model"
        prop_scorer.train(good_input, bad_input)
        prop_scorer.save_model("prop_scorer")
    
    # We need this for making sure that test sequences are not longer than what the trained model
    # expects.
    max_length = prop_scorer.scoring_model.get_input_shape_at(0)[1] 

    if args.test_file is not None:
        test_lines = [x.strip() for x in codecs.open(args.test_file,'r', 'utf-8').readlines()]
        num_test_propositions, test_indices = prop_scorer.prepare_test_data(test_lines, 
                max_length=max_length)
        print >>sys.stderr, "Scoring test data"
        test_all_prop_scores = prop_scorer.score(test_indices)
        test_scores = []
        t_ind = 0

        # Iterating through all propositions in the sentence and aggregating their scores
        for num_propositions in num_test_propositions:
            test_scores.append(test_all_prop_scores[t_ind:t_ind+num_propositions].sum())
            t_ind += num_propositions

        # Once aggregated, the number of scores should be the same as test sentences.
        assert len(test_scores) == len(test_lines)

        outfile = codecs.open("out.txt", "w", "utf-8")
        for score, line in zip(test_scores, test_lines):
            print >>outfile, score, line
        outfile.close()
