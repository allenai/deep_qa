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
from encoders import TreeCompositionLSTM

class NNSolver(object):
    def __init__(self):
        self.model = None
        self.data_indexer = DataIndexer()

    def train(self):
        # This function has to be implemented by the classes that inherit from 
        # this abstract class
        raise NotImplementedError

    def save_model(self, model_name_prefix, epoch):
        # Serializing the model for future use.
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json"%(model_name_prefix), "w")
        print >>model_config_file, model_config
        self.model.save_weights("%s_weights_epoch=%d.h5"%(model_name_prefix, epoch), overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        model_config_file.close()
        data_indexer_file.close()

    def load_model(self, model_name_prefix, epoch):
        # Loading serialized model
        model_config_file = open("%s_config.json"%(model_name_prefix))
        model_config_json = model_config_file.read()
        self.model = model_from_json(model_config_json, 
                custom_objects={"TreeCompositionLSTM": TreeCompositionLSTM})
        self.model.load_weights("%s_weights_epoch=%d.h5"%(model_name_prefix, epoch))
        data_indexer_file = open("%s_data_indexer.pkl"%model_name_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
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
        train_data_size = len(data_lines)

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
            training_sample = zip(good_elements, bad_elements)[:10]
            # Make int32 array so that Keras will view them as indices.
            good_elements = numpy.asarray(good_elements, dtype='int32')
            bad_elements = numpy.asarray(bad_elements, dtype='int32')
            # TreeLSTM's transitions input has an extra trailing dimension for 
            # concatenation. This has to match.
            transitions = numpy.expand_dims(transitions, axis=-1)
            num_good_inputs = good_elements.shape[0]
            num_bad_inputs = bad_elements.shape[0]
            # Labels will represent true statements as [0,1] and false statements as [1,0]
            labels = numpy.zeros((num_good_inputs+num_bad_inputs, 2))
            labels[:num_good_inputs, 1] = 1 # true statements
            labels[num_good_inputs:, 0] = 1 # false statements
            # Let's shuffle so that any sample taken from input-label combination will not have 
            # one dominating label.
            # Since transition sequences will not change for bad input, reusing those from
            # good input.
            zipped_input_labels = zip(numpy.concatenate([transitions, transitions]), 
                    numpy.concatenate([good_elements, bad_elements]), labels)
            random.shuffle(zipped_input_labels)
            shuffled_transitions, shuffled_elements, labels = [numpy.asarray(array) for 
                    array in zip(*zipped_input_labels)]
            inputs = (shuffled_transitions, shuffled_elements)
        else:
            # Padding to prespecified length
            good_input = self.data_indexer.pad_indices(good_input, max_length=max_length)
            bad_input = self.data_indexer.pad_indices(bad_input, max_length=max_length)
            #bad_input = self.data_indexer.corrupt_indices(good_input)
            training_sample = zip(good_input, bad_input)[:10]
            # Make int32 array so that Keras will view them as indices.
            good_input = numpy.asarray(good_input, dtype='int32')
            bad_input = numpy.asarray(bad_input, dtype='int32')
            inputs = numpy.concatenate([good_input, bad_input])
            num_good_inputs = good_input.shape[0]
            num_bad_inputs = bad_input.shape[0]
            print num_good_inputs, num_bad_inputs, train_data_size 
            # Labels will represent true statements as [0,1] and false statements as [1,0]
            labels = numpy.zeros((num_good_inputs+num_bad_inputs, 2))
            labels[:num_good_inputs, 1] = 1 # true statements
            labels[num_good_inputs:, 0] = 1 # false statements
            # Let's shuffle so that any sample taken from input-label combination will not have 
            # one dominating label.
            zipped_input_labels = zip(numpy.concatenate([good_input, bad_input]), labels)
            random.shuffle(zipped_input_labels)
            inputs, labels = [numpy.asarray(array) for array in zip(*zipped_input_labels)]
        
        # Print a training sample
        print >>sys.stderr, "Sample training pairs:"
        for good_prop_indices, bad_prop_indices in training_sample:
            print >>sys.stderr, "%s vs. %s"%(self.data_indexer.get_words_from_indices(good_prop_indices), 
                    self.data_indexer.get_words_from_indices(bad_prop_indices))

        return num_train_propositions, (inputs, labels)

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
        probabilities = self.model.predict(input)
        return probabilities[:, 1] # Return p(t|x) only

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

class LSTMSolver(NNSolver):
    def __init__(self):
        super(LSTMSolver, self).__init__()

    def train(self, train_input, train_labels, validation_input, validation_labels, 
            embedding_size=50, vocab_size=None, num_epochs=20):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices 
            from sentences in training data
        train_labels: numpy array: int32 (samples, 2). One hot representations of labels for 
            training
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

        ## STEP 1: Initialze the input layer
        input_layer = Input(shape=train_input.shape[1:], dtype='int32')

        ## STEP 2: Embed the propositions by changing word indices to word vectors
        # Mask zero ensures that padding is not a parameter in the entire model.
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        embed = embed_layer(input_layer)
        # Add a dropout to regularize the input representations
        regularized_embed = Dropout(0.5)(embed)

        ## STEP 3: Pass the sequences of word vectors through LSTM
        lstm_layer = LSTM(output_dim=embedding_size, W_regularizer=l2(0.01), 
                U_regularizer=l2(0.01), b_regularizer=l2(0.01), name='lstm')
        lstm_out = lstm_layer(regularized_embed)
        # Add a dropout after LSTM
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        ## STEP 4: Find p(true | proposition) by passing the outputs from LSTM through
        # an MLP with ReLU layers
        projection_layer = Dense(embedding_size/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        ## Step 5: Define crossentropy against labels as the loss and compile the model
        model = Model(input=input_layer, output=output_probabilities)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print >>sys.stderr, model.summary()
        self.model = model

        ## Step 6: Train the full model jointly
        best_accuracy = 0.0
        for epoch_id in range(num_epochs):
            # History callback contains the losses of all training samples
            print >>sys.stderr, "Epoch %d"%epoch_id
            history_callback = model.fit(train_input, train_labels, nb_epoch=1)
            accuracy = self.evaluate(validation_labels, validation_input)
            print >>sys.stderr, "Validation accuracy: %.4f"%accuracy
            if accuracy < best_accuracy:
                print >>sys.stderr, "Stopping training"
                break
            else:
                best_accuracy = accuracy
                self.save_model("propscorer_lstm", epoch_id)

class TreeLSTMSolver(NNSolver):
    def __init__(self):
        super(TreeLSTMSolver, self).__init__()

    def train(self, train_input, train_labels, validation_input, validation_labels, 
            embedding_size=50, vocab_size=None, num_epochs=20):
        '''
        train_input: List of two numpy arrays: transitions and initial buffer
        train_labels: numpy array (samples, 2): One hot label indices
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
        transitions, elements = train_input
        # Length of transitions (ops) is an upper limit on the stack and buffer
        # sizes. So use that to initialize the stack and buffer in the LSTM
        buffer_ops_limit = transitions.shape[1]
        stack_limit = buffer_ops_limit
        # The transitions input has an extra trailing dimension to make the 
        # concatenation with the buffer embedding easier.
        transitions_input = Input(shape=(buffer_ops_limit, 1))
        buffer_input = Input(shape=(buffer_ops_limit,), dtype='int32')

        ## STEP 2: Embed the propositions by changing word indices to word vectors
        # If merge in step 3 complains that input is masked, Keras needs to be updated. 
        # Merge supports masked input as of Keras 1.0.5
        embed_layer = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        embed = embed_layer(buffer_input)
        # Add a dropout to regularize the input representations
        regularized_embed = Dropout(0.5)(embed)

        ## STEP 3: Merge transitions and buffer (TreeLSTM only)
        lstm_input = merge([transitions_input, regularized_embed], 
                mode='concat')

        ## STEP 4: Pass the sequences of word vectors through TreeLSTM
        lstm_layer = TreeCompositionLSTM(stack_limit, buffer_ops_limit, 
                output_dim=embedding_size, W_regularizer=l2(0.01),
                U_regularizer=l2(0.01), V_regularizer=l2(0.01),
                b_regularizer=l2(0.01), name='treelstm')
        lstm_out = lstm_layer(lstm_input)
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        ## STEP 5: Find p(true | proposition) by passing the encoded proposition through
        # MLP with ReLU followed by softmax
        projection_layer = Dense(embedding_size/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        ## Step 6: Define crossentropy against labels as the loss and compile the model
        model = Model(input=[transitions_input, buffer_input], output=output_probabilities)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print >>sys.stderr, model.summary()
        self.model = model

        ## Step 7: Train the full model jointly
        best_accuracy = 0.0
        for epoch_id in range(num_epochs):
            print >>sys.stderr, "Epoch %d"%epoch_id
            history_callback = model.fit([transitions, elements], labels, nb_epoch=1)
            accuracy = self.evaluate(validation_labels, validation_input)
            print >>sys.stderr, "Validation accuracy: %.4f"%accuracy
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
    nn_solver = TreeLSTMSolver() if args.use_tree_lstm else LSTMSolver()

    if not args.train_file:
        # Training file is not given. There must be a serialized model.
        print >>sys.stderr, "Loading scoring model from disk"
        model_type = "treelstm" if args.use_tree_lstm else "lstm"
        model_name_prefix = "propscorer_%s"%model_type
        nn_solver.load_model(model_name_prefix, args.use_model_from_epoch)
        # input shape of scoring model is (samples, max_length)
    else:
        assert args.validation_file is not None, "Validation data is needed for training"
        print >>sys.stderr, "Reading training data"
        lines = [x.strip() for x in open(args.train_file).readlines()]
        _, (inputs, labels) = nn_solver.prepare_training_data(lines, 
                max_length=args.length_upper_limit, 
                in_shift_reduce_format=args.use_tree_lstm)

        train_sequence_length = inputs[0].shape[1] if args.use_tree_lstm \
                else inputs.shape[1]
        print >>sys.stderr, "Reading validation data"
        validation_lines = [x.strip() for x in codecs.open(args.validation_file,'r', 
            'utf-8').readlines()]
        validation_labels, validation_input = nn_solver.prepare_test_data(
                validation_lines, max_length=train_sequence_length, 
                in_shift_reduce_format=args.use_tree_lstm)
        if args.max_train_size is not None:
            print >>sys.stderr, "Limiting training size to %d"%(args.max_train_size)
            inputs = [input_[:args.max_train_size] for input_ in inputs] \
                    if args.use_tree_lstm else inputs[:args.max_train_size]
            labels = labels[:args.max_train_size]
        print >>sys.stderr, "Training model"
        nn_solver.train(inputs, labels, validation_input, validation_labels, 
                num_epochs=args.num_epochs)
    
    # We need this for making sure that test sequences are not longer than what the trained model
    # expects.
    max_length = nn_solver.model.get_input_shape_at(0)[0][1] \
            if args.use_tree_lstm else nn_solver.model.get_input_shape_at(0)[1]

    if args.test_file is not None:
        test_lines = [x.strip() for x in codecs.open(args.test_file,'r', 'utf-8').readlines()]
        test_labels, test_input = nn_solver.prepare_test_data(test_lines, 
                max_length=max_length, in_shift_reduce_format=args.use_tree_lstm)
        print >>sys.stderr, "Scoring test data"
        test_scores = nn_solver.score(test_input)
        accuracy = nn_solver.evaluate(test_labels, test_input)
        print >>sys.stderr, "Test accuracy: %.4f"%accuracy
        assert len(test_scores) == len(test_lines)

        outfile = codecs.open("out.txt", "w", "utf-8")
        for score, line in zip(test_scores, test_lines):
            print >>outfile, score, line
        outfile.close()
