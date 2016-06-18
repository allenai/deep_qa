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
from treecomp_lstm import TreeCompositionLSTM

class PropScorer(object):
    def __init__(self):
        self.model = None
        self.data_indexer = DataIndexer()

    def train(self, good_input, bad_input, embedding_size=50, vocab_size=None, 
            use_tree_lstm=False):
        '''
        good_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from sentences in training data
        bad_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from corrupted versions of sentences in training data
        embedding_size: int. Size of word vectors
        vocab_size: int. Input dimensionality of embedding layer. Will be inferred from inputs 
            if not provided.
        use_tree_lstm: bool. Use treeLSTM composition instead of processing input as 
            sequences
        '''
        if vocab_size is None:
            vocab_size = self.data_indexer.get_vocab_size()

        ## STEP 1: Initialze the two inputs
        if use_tree_lstm:
            good_transitions, good_elements = good_input
            bad_transitions, bad_elements = bad_input
            # Length of transitions (ops) is an upper limit on the stack and buffer
            # sizes. So use that to initialize the stack and buffer in the LSTM
            buffer_ops_limit = good_transitions.shape[1]
            stack_limit = buffer_ops_limit
            # The transitions input has an extra trailing dimension to make the 
            # concatenation with the buffer embedding easier.
            good_transitions_input = Input(shape=(buffer_ops_limit, 1))
            good_buffer_input = Input(shape=(buffer_ops_limit,), dtype='int32')
            bad_transitions_input = Input(shape=(buffer_ops_limit, 1))
            bad_buffer_input = Input(shape=(buffer_ops_limit,), dtype='int32')
        else:
            good_input_layer = Input(shape=good_input.shape[1:], dtype='int32')
            bad_input_layer = Input(shape=bad_input.shape[1:], dtype='int32')

        ## STEP 2: Embed the two propositions by changing word indices to word vectors
        # Share embedding layer for both propositions
        if use_tree_lstm:
            # We are not masking zero here since the merge in step 3 complains if input
            # is masked. However, masking is unnecessary because the TreeLSTM's states
            # do not change when the transition is not one of the three ops, making the
            # gradient 0.
            # TODO (pradeep): Confirm the claim.
            embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                    name='embedding')
            good_embed = embed_layer(good_buffer_input)
            bad_embed = embed_layer(bad_buffer_input)
        else:
            # Mask zero ensures that padding is not a parameter in the entire model.
            embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                    mask_zero=True, name='embedding')
            good_embed = embed_layer(good_input_layer)
            bad_embed = embed_layer(bad_input_layer)
        # Add a dropout to regularize the input representations
        regularized_good_embed = Dropout(0.5)(good_embed)
        regularized_bad_embed = Dropout(0.5)(bad_embed)

        ## STEP 3: Merge transitions and buffer (TreeLSTM only)
        if use_tree_lstm:
            good_lstm_input = merge([good_transitions_input, regularized_good_embed], 
                    mode='concat')
            bad_lstm_input = merge([bad_transitions_input, regularized_bad_embed], 
                    mode='concat')

        ## STEP 4: Pass the sequences of word vectors through the same LSTM
        if use_tree_lstm:
            lstm_layer = TreeCompositionLSTM(stack_limit, buffer_ops_limit, 
                    output_dim=embedding_size, W_regularizer=l2(0.01),
                    U_regularizer=l2(0.01), V_regularizer=l2(0.01),
                    b_regularizer=l2(0.01), name='treelstm')
            good_lstm_out = lstm_layer(good_lstm_input)
            bad_lstm_out = lstm_layer(bad_lstm_input)
        else:
            lstm_layer = LSTM(output_dim=embedding_size, W_regularizer=l2(0.01), 
                    U_regularizer=l2(0.01), b_regularizer=l2(0.01), name='lstm')
        # Share LSTM layer for both propositions
            good_lstm_out = lstm_layer(regularized_good_embed)
            bad_lstm_out = lstm_layer(regularized_bad_embed)
	regularized_good_lstm_out = Dropout(0.2)(good_lstm_out)
	regularized_bad_lstm_out = Dropout(0.2)(bad_lstm_out)

        ## STEP 5: Score the two propositions by passing the outputs from LSTM twough the same 
        # dense layer
        #TODO: Can make the scorer more complex by adding more layers
        scorer_layer = Dense(1, activation='tanh', W_regularizer=l2(0.01), b_regularizer=l2(0.01), name='scorer')
        # Share scoring layer for both propositions
        good_score = scorer_layer(regularized_good_lstm_out)
        bad_score = scorer_layer(regularized_bad_lstm_out)

        ## Step 6: Define the score difference as the loss and compile the model
        score_diff = merge([good_score, bad_score], mode=lambda scores: scores[1] - scores[0], 
                output_shape=lambda shapes: shapes[0])
        # Defining a simple hingeloss that depends only on the diff between scores of the two 
        # inputs, since Keras has only supervised losses predefined.
        # But Keras expects a "target" (dummy_target) in a loss. Just ignore it in the function. 
        # Technically dummy_target is an argument of the final function, and Theano will complain 
        # if it is not a part of the computational graph. So, resorting to this hack of 0*dummy_target
        # Loss = max(0, margin + bad_score - good_score)
        margin = 0.5
        score_hinge_loss = lambda dummy_target, diff: K.mean(K.switch(margin+diff>0.0, margin+diff, 0.0) + 0*dummy_target, axis=-1)
        if use_tree_lstm:
            model = Model(input=[good_transitions_input, good_buffer_input, 
                bad_transitions_input, bad_buffer_input], output=score_diff)
        else:
            model = Model(input=[good_input_layer, bad_input_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='adam')
        print >>sys.stderr, model.summary()

        ## Step 7: Train the full model jointly
        # Define early stopping with patience of 1 (will wait for atmost one epoch after validation loss 
        # stops decreasing). The quantity being monitored is the validation loss (default option).
        early_stopping = EarlyStopping(patience=1)
        # Separate the last 10% of training data as validation data, which is used for early stopping
        # Note: target is ignored by loss. See above.
        if use_tree_lstm:
            model.fit([good_transitions, good_elements, bad_transitions, bad_elements], 
                    numpy.zeros((good_input[1].shape[:1])), validation_split=0.1, 
                    callbacks=[early_stopping]) 
        else:
            model.fit([good_input, bad_input], numpy.zeros((good_input.shape[:1])), 
                    validation_split=0.1, callbacks=[early_stopping]) 
        self.model = model

        ## Step 8: Define a scoring model taking the necessary parts of the trained model
        if use_tree_lstm:
            test_transitions_input = Input(shape=(buffer_ops_limit, 1))
            test_buffer_input = Input(shape=(buffer_ops_limit,), dtype='int32')
            test_buffer_embed = embed_layer(test_buffer_input)
            test_lstm_out = lstm_layer(merge([test_transitions_input, 
                test_buffer_embed], mode='concat'))
            test_score = scorer_layer(test_lstm_out)
            self.scoring_model = Model(input=[test_transitions_input, test_buffer_input], 
                    output=test_score)
        else:
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
        self.scoring_model = model_from_json(model_config_json, 
                custom_objects={"TreeCompositionLSTM": TreeCompositionLSTM})
        self.scoring_model.load_weights("%s_weights.h5"%model_name_prefix)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.scoring_model.compile(loss='mse', optimizer='sgd')
        model_config_file.close()
        data_indexer_file.close()

    def prepare_training_data(self, data_lines, max_length=None, 
            in_shift_reduce_format=False):
        # max_length is used for ignoring long training instances
        # and also padding all instances to make them of the same 
        # length

        # Indexing training data
        print >>sys.stderr, "Indexing training data"
        num_train_propositions, good_input = self.data_indexer.process_data(data_lines, 
                separate_propositions=False, for_train=True, max_length=max_length)
        # Corrupting train indices to get "bad" data
        print >>sys.stderr, "Corrupting training data"
        if in_shift_reduce_format:
            transitions, good_elements = self.data_indexer.get_shift_reduce_sequences(
                    good_input)
            transitions = self.data_indexer.pad_indices(transitions)
            # Insight: Transition sequences are strictly longer than element sequences
            # because there will be exactly as many shifts as there are elements, and
            # then some reduce operations on top of that
            # So we can safely use transitions' max length for elements as well. TreeLSTM
            # implementation requires them to be of the same length for concatenation.
            max_transition_length = len(transitions[0])
            good_elements = self.data_indexer.pad_indices(good_elements, 
                    max_length=max_transition_length)
            bad_elements = self.data_indexer.corrupt_indices(good_elements)
            training_sample = zip(good_elements, bad_elements)[:10]
            # Make int32 array so that Keras will view them as indices.
            good_elements = numpy.asarray(good_elements, dtype='int32')
            bad_elements = numpy.asarray(bad_elements, dtype='int32')
            # TreeLSTM's transitions input has an extra trailing dimension for 
            # concatenation. This has to match.
            transitions = numpy.expand_dims(transitions, axis=-1)
            good_input = (transitions, good_elements)
            # Since transition sequences will not change for bad input, reusing those from
            # good input.
            bad_input = (transitions, bad_elements)
        else:
            # Padding to prespecified length
            good_input = self.data_indexer.pad_indices(good_input, max_length=max_length)
            bad_input = self.data_indexer.corrupt_indices(good_input)
            training_sample = zip(good_input, bad_input)[:10]
            # Make int32 array so that Keras will view them as indices.
            good_input = numpy.asarray(good_input, dtype='int32')
            bad_input = numpy.asarray(bad_input, dtype='int32')

        # Print a training sample
        print >>sys.stderr, "Sample training pairs:"
        for good_prop_indices, bad_prop_indices in training_sample:
            print >>sys.stderr, "%s vs. %s"%(self.data_indexer.get_words_from_indices(good_prop_indices), 
                    self.data_indexer.get_words_from_indices(bad_prop_indices))

        return num_train_propositions, (good_input, bad_input)

    def prepare_test_data(self, data_lines, max_length=None, 
            in_shift_reduce_format=False):
        # Indexing test data
        print >>sys.stderr, "Indexing test data"
        num_test_propositions, test_indices = self.data_indexer.process_data(test_lines, 
                for_train=False)
        if in_shift_reduce_format:
            test_transitions, test_elements = self.data_indexer.get_shift_reduce_sequences(
                    test_indices)
            test_transitions = self.data_indexer.pad_indices(test_transitions, 
                    max_length=max_length)
            test_elements = self.data_indexer.pad_indices(test_elements, 
                    max_length=max_length)
            # Make int32 array so that Keras will view them as indices.
            test_elements = numpy.asarray(test_elements, dtype='int32')
            # Adding trailing dimension to match TreeLSTM's input
            test_transitions = numpy.expand_dims(test_transitions, axis=-1)
            # The following needs to be a list for Keras to process them as inputs
            test_input = [test_transitions, test_elements]
        else:
            test_indices = self.data_indexer.pad_indices(test_indices, 
                    max_length=max_length)
            # Make int32 array so that Keras will view them as indices.
            test_input = numpy.asarray(test_indices, dtype='int32')
        return num_test_propositions, test_input
        
    def score(self, input):
        score_val = self.scoring_model.predict(input)
        return score_val

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--use_tree_lstm', help="Use TreeLSTM composition", 
            action='store_true')
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
                max_length=args.length_upper_limit, 
                in_shift_reduce_format=args.use_tree_lstm)

        if args.max_train_size is not None:
            print >>sys.stderr, "Limiting training size to %d"%(args.max_train_size)
            good_input = good_input[:args.max_train_size]
            bad_input = bad_input[:args.max_train_size]
        print >>sys.stderr, "Training model"
        prop_scorer.train(good_input, bad_input, use_tree_lstm=args.use_tree_lstm)
        prop_scorer.save_model("prop_scorer")
    
    # We need this for making sure that test sequences are not longer than what the trained model
    # expects.
    max_length = prop_scorer.scoring_model.get_input_shape_at(0)[0][1]

    if args.test_file is not None:
        test_lines = [x.strip() for x in codecs.open(args.test_file,'r', 'utf-8').readlines()]
        num_test_propositions, test_input = prop_scorer.prepare_test_data(test_lines, 
                max_length=max_length, in_shift_reduce_format=args.use_tree_lstm)
        print >>sys.stderr, "Scoring test data"
        test_all_prop_scores = prop_scorer.score(test_input)
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
