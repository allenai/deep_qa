import sys
import argparse
import random
import numpy
import codecs
import pickle
from keras.layers import Embedding, Input, LSTM, Dense, Dropout, merge
from keras.models import Model, model_from_json
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l2
from keras import backend as K

from index_data import DataIndexer
from treecomp_lstm import TreeCompositionLSTM

class NNScorer(object):
    def __init__(self):
        self.model = None
        self.data_indexer = DataIndexer()

    def train(self):
        # This function has to be implemented by the classes that inherit from 
        # this abstract class
        raise NotImplementedError

    def save_model(self, model_name_prefix, epoch):
        # Serializing the scoring model for future use.
        model_config = self.scoring_model.to_json()
        model_config_file = open("%s_config.json"%(model_name_prefix), "w")
        print >>model_config_file, model_config
        self.scoring_model.save_weights("%s_weights_epoch=%d.h5"%(model_name_prefix, epoch), overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        model_config_file.close()
        data_indexer_file.close()

    def load_model(self, model_name_prefix, epoch):
        # Loading serialized model
        model_config_file = open("%s_config.json"%(model_name_prefix))
        model_config_json = model_config_file.read()
        self.scoring_model = model_from_json(model_config_json, 
                custom_objects={"TreeCompositionLSTM": TreeCompositionLSTM})
        self.scoring_model.load_weights("%s_weights_epoch=%d.h5"%(model_name_prefix, epoch))
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
        
        negative_samples_given = False
        # If the the input file has two columns, we assume the second column has 
        # negative samples.
        # TODO: This seems fragile. More checks needed.
        if "\t" in data_lines[0]:
            good_lines, bad_lines = zip(*[line.split("\t") for line in data_lines])
            negative_samples_given = True
            print >>sys.stderr, "Negative training data provided."
        else:
            good_lines = data_lines

        # Indexing training data
        print >>sys.stderr, "Indexing training data"
        num_train_propositions, good_input = self.data_indexer.process_data(good_lines, 
                separate_propositions=False, for_train=True, max_length=max_length)
        if negative_samples_given:
            _, bad_input = self.data_indexer.process_data(bad_lines, 
                    separate_propositions=False, for_train=True, max_length=max_length)
        else:
            # Corrupting train indices to get "bad" data
            print >>sys.stderr, "Corrupting training data"
            bad_input = self.data_indexer.corrupt_indices(good_input)
        if in_shift_reduce_format:
            transitions, good_elements = self.data_indexer.get_shift_reduce_sequences(
                    good_input)
            # Assuming the transitions will be the same for both good and bad inputs
            # because the structure is not different.
            _, bad_elements = self.data_indexer.get_shift_reduce_sequences(
                    bad_input)
            transitions = self.data_indexer.pad_indices(transitions)
            # Insight: Transition sequences are strictly longer than element sequences
            # because there will be exactly as many shifts as there are elements, and
            # then some reduce operations on top of that
            # So we can safely use transitions' max length for elements as well. TreeLSTM
            # implementation requires them to be of the same length for concatenation.
            max_transition_length = len(transitions[0])
            good_elements = self.data_indexer.pad_indices(good_elements, 
                    max_length=max_transition_length)
            bad_elements = self.data_indexer.pad_indices(bad_elements, 
                    max_length=max_transition_length)
            #bad_elements = self.data_indexer.corrupt_indices(good_elements)
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
            bad_input = self.data_indexer.pad_indices(bad_input, max_length=max_length)
            #bad_input = self.data_indexer.corrupt_indices(good_input)
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
        # data_lines expected to be in tab separated format with first column being
        # 1 or 0 indicatind correct answers and the second column being the logical form
        test_answer_strings, test_lines = zip(*[line.split("\t") for line in data_lines])
        assert len(test_answer_strings)%4 == 0, "Not enough lines per question"
        num_questions = len(test_answer_strings) / 4
        print >>sys.stderr, "Read %d questions"%num_questions
        test_answers = numpy.asarray([int(x) for x in test_answer_strings]).reshape(num_questions, 4)
        assert numpy.all(numpy.asarray([numpy.count_nonzero(ta) for ta in 
            test_answers]) == 1), "Some questions do not have exactly one correct answer"
        test_labels = numpy.argmax(test_answers, axis=1)

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
        return test_labels, test_input
        
    def score(self, input):
        score_val = self.scoring_model.predict(input)
        return score_val

    def evaluate(self, labels, test_input):
        # labels: list(int). List of indices of correct answers
        # test_input: list(numpy arrays) or numpy array: input to score
        test_scores = self.score(test_input)
        num_questions = len(labels)
        # test_scores will be of length 4*num_questions. Let's reshape it
        # to a matrix of size (num_questions, 4) and take argmax of over 
        # the four options for each question.
        test_predictions = numpy.argmax(test_scores.reshape(num_questions, 4), 
                axis=1)
        num_correct = sum(test_predictions == labels)
        accuracy = float(num_correct)/num_questions
        return accuracy

class LSTMScorer(NNScorer):
    def __init__(self):
        super(LSTMScorer, self).__init__()

    def train(self, good_input, bad_input, validation_input, validation_labels, 
            embedding_size=50, vocab_size=None, num_epochs=20):
        '''
        good_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from sentences in training data
        bad_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from corrupted versions of sentences in training data
        validation_input: numpy array: int32 (4*num_valid_questions, num_words). Array 
            similar to good and bad inputs. Every set of rows correspond to the statements 
            formed from four options of a question.
        validation_labels: numpy array: int32 (num_valid_questions,). Indices of correct 
            answers
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
        # Share embedding layer for both propositions
        # Mask zero ensures that padding is not a parameter in the entire model.
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        good_embed = embed_layer(good_input_layer)
        bad_embed = embed_layer(bad_input_layer)
        # Add a dropout to regularize the input representations
        regularized_good_embed = Dropout(0.5)(good_embed)
        regularized_bad_embed = Dropout(0.5)(bad_embed)

        ## STEP 3: Pass the sequences of word vectors through the same LSTM
        lstm_layer = LSTM(output_dim=embedding_size, W_regularizer=l2(0.01), 
                U_regularizer=l2(0.01), b_regularizer=l2(0.01), name='lstm')
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
        # Technically dummy_target is an argument of the final function, and Theano will complain 
        # if it is not a part of the computational graph. So, resorting to this hack of 0*dummy_target
        # Loss = max(0, margin + bad_score - good_score)
        margin = 0.5
        score_hinge_loss = lambda dummy_target, diff: K.mean(K.switch(margin+diff>0.0, margin+diff, 0.0) + 
                0*dummy_target, axis=-1)
        model = Model(input=[good_input_layer, bad_input_layer], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='adam')
        print >>sys.stderr, model.summary()

        ## Step 6: Define and compile the scoring model
        # The layers in this model are shared with the full model. So they will change as the full model
        # is trained.
        test_input = Input(good_input.shape[1:], dtype='int32')
        test_embed = embed_layer(test_input)
        test_lstm_out = lstm_layer(test_embed)
        test_score = scorer_layer(test_lstm_out)
        self.scoring_model = Model(input=test_input, output=test_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not 
        # going to train.
        self.scoring_model.compile(loss='mse', optimizer='sgd')

        ## Step 7: Train the full model jointly
        inputs = [good_input, bad_input]
        targets = numpy.zeros((good_input.shape[:1]))
        # Note: target is ignored by loss. See above.
        
        best_accuracy = 0.0
        for epoch_id in range(num_epochs):
            # History callback contains the losses of all training samples
            history_callback = model.fit(inputs, targets, nb_epoch=1)
            training_loss = numpy.mean(history_callback.history['loss']) 
            accuracy = self.evaluate(validation_labels, validation_input)
            print >>sys.stderr, "Validation accuracy: %.4f"%accuracy
            if training_loss < margin:
                # We want to either stop training or update the best accuracy
                # only if the training loss went below the margin. If not, 
                # the model hasn't learned anything yet.
                if accuracy < best_accuracy:
                    print >>sys.stderr, "Stopping training"
                    break
                else:
                    best_accuracy = accuracy
                    self.save_model("propscorer_lstm", epoch_id)

class TreeLSTMScorer(NNScorer):
    def __init(self):
        super(TreeLSTMScorer, self).__init__()

    def train(self, good_input, bad_input, validation_input, validation_labels, 
            embedding_size=50, vocab_size=None, num_epochs=20):
        '''
        good_input: List of two numpy arrays: transitions and good initial buffer
        bad_input: List of two numpy arrays: transitions and bad initial buffer
        validation_input: List similar to good and bad inputs. Every set of rows in 
            both arrays correspond to the statements formed from four options of a question.
        validation_labels: numpy array: int32 (num_valid_questions,). Indices of correct 
            answers
        embedding_size: int. Size of word vectors
        vocab_size: int. Input dimensionality of embedding layer. Will be inferred from inputs 
            if not provided.
        '''
        if vocab_size is None:
            vocab_size = self.data_indexer.get_vocab_size()

        ## STEP 1: Initialze the two inputs
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

        ## STEP 2: Embed the two propositions by changing word indices to word vectors
        # Share embedding layer for both propositions
        # If merge in step 3 complains that input is masked, Keras needs to be updated. 
        # Merge supports masked input as of Keras 1.0.5
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        good_embed = embed_layer(good_buffer_input)
        bad_embed = embed_layer(bad_buffer_input)
        # Add a dropout to regularize the input representations
        regularized_good_embed = Dropout(0.5)(good_embed)
        regularized_bad_embed = Dropout(0.5)(bad_embed)

        ## STEP 3: Merge transitions and buffer (TreeLSTM only)
        good_lstm_input = merge([good_transitions_input, regularized_good_embed], 
                mode='concat')
        bad_lstm_input = merge([bad_transitions_input, regularized_bad_embed], 
                mode='concat')

        ## STEP 4: Pass the sequences of word vectors through the same LSTM
        lstm_layer = TreeCompositionLSTM(stack_limit, buffer_ops_limit, 
                output_dim=embedding_size, W_regularizer=l2(0.01),
                U_regularizer=l2(0.01), V_regularizer=l2(0.01),
                b_regularizer=l2(0.01), name='treelstm')
        # Share LSTM layer for both propositions
        good_lstm_out = lstm_layer(good_lstm_input)
        bad_lstm_out = lstm_layer(bad_lstm_input)
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
        score_hinge_loss = lambda dummy_target, diff: K.mean(K.switch(margin+diff>0.0, margin+diff, 0.0) + 
                0*dummy_target, axis=-1)
        model = Model(input=[good_transitions_input, good_buffer_input, 
            bad_transitions_input, bad_buffer_input], output=score_diff)
        model.compile(loss=score_hinge_loss, optimizer='adam')
        print >>sys.stderr, model.summary()

        ## Step 7: Define and compile the scoring model
        # The layers in this model are shared with the full model. So they will change as the full model
        # is trained.
        test_transitions_input = Input(shape=good_input[0].shape[1:])
        test_buffer_input = Input(shape=good_input[1].shape[1:], dtype='int32')
        test_buffer_embed = embed_layer(test_buffer_input)
        test_lstm_out = lstm_layer(merge([test_transitions_input, 
            test_buffer_embed], mode='concat'))
        test_score = scorer_layer(test_lstm_out)
        self.scoring_model = Model(input=[test_transitions_input, test_buffer_input], 
                output=test_score)
        # We just need the model to predict. The loss and optimizer do not matter since we are not 
        # going to train.
        self.scoring_model.compile(loss='mse', optimizer='sgd')

        ## Step 8: Train the full model jointly
        inputs = [good_transitions, good_elements, bad_transitions, bad_elements]
        targets = numpy.zeros((good_input[1].shape[:1]))
        # Note: target is ignored by loss. See above.
        
        best_accuracy = 0.0
        for epoch_id in range(num_epochs):
            history_callback = model.fit(inputs, targets, nb_epoch=1)
            training_loss = numpy.mean(history_callback.history['loss']) 
            accuracy = self.evaluate(validation_labels, validation_input)
            print >>sys.stderr, "Validation accuracy: %.4f"%accuracy
            if training_loss < margin:
                # We want to either stop training or update the best accuracy
                # only if the training loss went below the margin. If not, 
                # the model hasn't learned anything yet.
                if accuracy < best_accuracy:
                    print >>sys.stderr, "Stopping training"
                    break
                else:
                    best_accuracy = accuracy
                    self.save_model("propscorer_treelstm", epoch_id)

if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('--train_file', type=str)
    argparser.add_argument('--validation_file', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--use_tree_lstm', help="Use TreeLSTM composition", 
            action='store_true')
    argparser.add_argument('--length_upper_limit', type=int, 
            help="Upper limit on length of training data. Ignored during testing.")
    argparser.add_argument('--max_train_size', type=int, 
            help="Upper limit on the size of training data")
    argparser.add_argument('--num_epochs', type=int, default=20,
            help="Number of train epochs (20 by default)")
    argparser.add_argument('--use_model_from_epoch', type=int, default=0, 
            help="Use the model from a particular epoch (0 by default)")
    args = argparser.parse_args()
    prop_scorer = TreeLSTMScorer() if args.use_tree_lstm else LSTMScorer()

    if not args.train_file:
        # Training file is not given. There must be a serialized model.
        print >>sys.stderr, "Loading scoring model from disk"
        model_type = "treelstm" if args.use_tree_lstm else "lstm"
        model_name_prefix = "propscorer_%s"%model_type
        prop_scorer.load_model(model_name_prefix, args.use_model_from_epoch)
        # input shape of scoring model is (samples, max_length)
    else:
        assert args.validation_file is not None, "Validation data is needed for training"
        print >>sys.stderr, "Reading training data"
        lines = [x.strip() for x in open(args.train_file).readlines()]
        # Shuffling lines now since Keras does not shuffle data before validation
        # split is made. This will ensure validation data is not at the end of the file.
        random.shuffle(lines)
        _, (good_input, bad_input) = prop_scorer.prepare_training_data(lines, 
                max_length=args.length_upper_limit, 
                in_shift_reduce_format=args.use_tree_lstm)

        train_sequence_length = good_input[0].shape[1] if args.use_tree_lstm \
                else good_input.shape[1]
        print >>sys.stderr, "Reading validation data"
        validation_lines = [x.strip() for x in codecs.open(args.validation_file,'r', 
            'utf-8').readlines()]
        validation_labels, validation_input = prop_scorer.prepare_test_data(
                validation_lines, max_length=train_sequence_length, 
                in_shift_reduce_format=args.use_tree_lstm)
        if args.max_train_size is not None:
            print >>sys.stderr, "Limiting training size to %d"%(args.max_train_size)
            good_input = [input_[:args.max_train_size] for input_ in good_input] \
                    if args.use_tree_lstm else good_input[:args.max_train_size]
            bad_input = [input_[:args.max_train_size] for input_ in bad_input] \
                    if args.use_tree_lstm else bad_input[:args.max_train_size]
        print >>sys.stderr, "Training model"
        prop_scorer.train(good_input, bad_input, validation_input, validation_labels, 
                num_epochs=args.num_epochs)
    
    # We need this for making sure that test sequences are not longer than what the trained model
    # expects.
    max_length = prop_scorer.scoring_model.get_input_shape_at(0)[0][1] \
            if args.use_tree_lstm else prop_scorer.scoring_model.get_input_shape_at(0)[1]

    if args.test_file is not None:
        test_lines = [x.strip() for x in codecs.open(args.test_file,'r', 'utf-8').readlines()]
        test_labels, test_input = prop_scorer.prepare_test_data(test_lines, 
                max_length=max_length, in_shift_reduce_format=args.use_tree_lstm)
        print >>sys.stderr, "Scoring test data"
        test_scores = prop_scorer.score(test_input)
        accuracy = prop_scorer.evaluate(test_labels, test_input)
        print >>sys.stderr, "Test accuracy: %.4f"%accuracy
        assert len(test_scores) == len(test_lines)

        outfile = codecs.open("out.txt", "w", "utf-8")
        for score, line in zip(test_scores, test_lines):
            print >>outfile, score, line
        outfile.close()
