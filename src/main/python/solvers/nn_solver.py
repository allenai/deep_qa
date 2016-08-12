from __future__ import print_function
import sys
import argparse
import codecs
import pickle

import numpy
from keras.layers import Embedding, Input, LSTM, Dense, Dropout, merge
from keras.models import Model, model_from_json
from keras.regularizers import l2

from ..data.dataset import Dataset, IndexedDataset
from ..data.index_data import DataIndexer
from ..layers.encoders import TreeCompositionLSTM

class NNSolver(object):
    def __init__(self, model_prefix, **kwargs):
        """
        model_prefix: specifies where to save or load model files.

        Allowed kwargs:

        embedding_size: int. Size of word vectors (default 50).
        max_sentence_length: max length of training sentences (ignored at test time).

        train_file: path to training data.
        positive_train_file: path to positive training data.
        negative_train_file: path to negative training data.
        validation_file: path to validation data.

        NOTE on train file arguments: if `train_file` is given, the other two arguments are
        ignored, and the file is assumed to have instance labels.  If `positive_train_file` is
        given, it is assumed to not have labels (or all labels must be "1").  Similarly for
        `negative_train_file`, except label must be "0" if present.  If `positive_train_file` is
        given and `negative_train_file` isn't, we will try to generate negative data, but the
        method to do so is poor and won't work for all subclasses.
        """
        self.model_prefix = model_prefix
        self.embedding_size = kwargs.get('embedding_size', 50)
        self.max_sentence_length = kwargs.get('max_sentence_length', None)
        self.train_file = kwargs.get('train_file', None)
        self.positive_train_file = kwargs.get('positive_train_file', None)
        self.negative_train_file = kwargs.get('negative_train_file', None)
        self.validation_file = kwargs.get('validation_file', None)

        self.data_indexer = DataIndexer()
        self.model = None
        self.best_epoch = -1

    def set_max_sentence_length_from_model(self):
        """
        Given a loaded model, set the max_sentence_length, so we know what length to pad data to
        when using this loaded model.
        """
        raise NotImplementedError

    def get_simple_training_input(self) -> IndexedDataset:
        """
        If your NNSolver subclass only has sentence/logical form files as input (i.e., no
        background or other metadata associated with inputs in other files), and that you
        instantiated the NNSolver by passing the filenames in kwargs, we will try to load the data
        here.  If you need something more complicated (like with the memory network), you'll have
        to write code to load the data yourself (or use this and then do extra stuff too).
        """
        if self.train_file:
            dataset = Dataset.read_from_file(self.train_file)
        else:
            positive_dataset = Dataset.read_from_file(self.positive_train_file, label=True)
            if self.negative_train_file:
                negative_dataset = Dataset.read_from_file(self.negative_train_file, label=False)
                dataset = positive_dataset.merge(negative_dataset)
        self.data_indexer.fit_word_dictionary(dataset)
        indexed_dataset = self.data_indexer.index_dataset(dataset)
        if self.positive_train_file and not self.negative_train_file:
            corrupted_dataset = self.data_indexer.corrupt_dataset(indexed_dataset)
            indexed_dataset = indexed_dataset.merge(corrupted_dataset)
        return indexed_dataset

    def get_training_data(self):
        """Loads training data and converts it into a format suitable for input to Keras.  This
        method must return a tuple of (train_input, train_labels).

        This method takes no arguments; any necessary arguments (e.g., a path for where to find the
        training data) must have been passed to the constructor of this object.
        """
        raise NotImplementedError

    def get_validation_data(self):
        """Like get_training_data, but for validation data.
        """
        raise NotImplementedError

    def build_model(self, train_input, vocab_size):
        """Constructs and returns a Keras model that will take train_input as input, and produce as
        output a true/false decision for each input.

        The returned model will be used to call model.fit(train_input, train_labels).
        """
        raise NotImplementedError

    def train(self, **kwargs):
        # TODO(matt): make methods that create train_input, train_labels, validation_input, and
        # validation_labels, so that each subclass can do this on its own.  This is basically just
        # cleaning up the API of the prepare_data methods that already exist.
        # TODO(matt): as part of that API, make an "embed_word_sequence_inputs" method, that either
        # adds an Embedding layer, or uses a fixed word2vec embedding then a projection, etc.
        '''
        train_input: Exact format depends on subclass, but this will be passed directly to
            model.fit().
        train_labels: numpy array: int32 (samples, 2). One hot representations of labels for
            training
        validation_input: Similar to train_input, except it must have length that is a multiple of
            four.   Every set of four rows correspond to the statements formed from four options of
            a question.
        validation_labels: numpy array: int32 (num_valid_questions,). Indices of correct
            answers

        Allowed kwargs:
        num_epochs: int. Number of training epochs (default 20).
        patience: int. Number of patience epochs for deciding on early stopping (default 1).
        '''
        num_epochs = kwargs.get('num_epochs', 20)
        patience = kwargs.get('patience', 1)

        # First we need to prepare the data that we'll use for training.
        train_input, train_labels = self.get_training_data()
        validation_input, validation_labels = self.get_validation_data()

        # This must be called after self.get_training_data(), because that's where we determine the
        # vocabulary size.
        vocab_size = self.data_indexer.get_vocab_size()

        # Then we build the model.  This creates a compiled Keras Model.
        self.model = self.build_model(train_input, vocab_size)

        # Now we actually train the model, with patient early stopping using the validation data.
        best_accuracy = 0.0
        self.best_epoch = 0
        num_worse_epochs = 0
        for epoch_id in range(num_epochs):
            print("Epoch %d" % epoch_id, file=sys.stderr)
            self.model.fit(train_input, train_labels, nb_epoch=1)
            accuracy = self.evaluate(validation_labels, validation_input)
            print("Validation accuracy: %.4f" % accuracy, file=sys.stderr)
            if accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= patience:
                    print("Stopping training", file=sys.stderr)
                    break
            else:
                best_accuracy = accuracy
                self.best_epoch = epoch_id
                self.save_model(epoch_id)
        self.save_best_model()

    def save_model(self, epoch):
        # Serializing the model for future use.
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json" % (self.model_prefix), "w")
        print(model_config, file=model_config_file)
        self.model.save_weights("%s_weights_epoch=%d.h5" % (self.model_prefix, epoch), overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        model_config_file.close()
        data_indexer_file.close()

    def save_best_model(self):
        '''Copies the weights from the best epoch to a final weight file

        The point of this is so that the input/output spec of the NNSolver is simpler.  Someone
        calling this as a subroutine doesn't have to worry about which epoch ended up being the
        best, they can just use the final weight file.  You can still use models from other epochs
        if you really want to.
        '''
        from shutil import copyfile
        epoch_weight_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, self.best_epoch)
        final_weight_file = "%s_weights.h5" % self.model_prefix
        copyfile(epoch_weight_file, final_weight_file)

    def load_model(self, epoch=None, custom_objects=None):
        # Loading serialized model
        model_config_file = open("%s_config.json" % self.model_prefix)
        model_config_json = model_config_file.read()
        self.model = model_from_json(model_config_json, custom_objects=custom_objects)
        if epoch is not None:
            model_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, epoch)
        else:
            model_file = "%s_weights.h5" % self.model_prefix
        self.model.load_weights(model_file)
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        model_config_file.close()
        data_indexer_file.close()
        self.set_max_sentence_length_from_model()

    def prepare_test_data(self, data_lines, max_length=None, in_shift_reduce_format=False):
        # data_lines expected to be in tab separated format, either two or three columns, formatted
        # as: "[sentence][tab][label]" or "[sentence index][tab][sentence][tab][label]".  Label is
        # either 0 or 1.
        num_fields = len(data_lines[0].split("\t"))
        if num_fields == 2:
            test_lines, test_label_strings = zip(*[line.split("\t") for line in data_lines])
        else:
            _, test_lines, test_label_strings = zip(*[line.split("\t") for line in data_lines])
        assert len(test_label_strings)%4 == 0, "Not enough lines per question"
        num_questions = len(test_label_strings) / 4
        print("Read %d questions" % num_questions, file=sys.stderr)
        test_answers = numpy.asarray([int(x) for x in test_label_strings]).reshape(num_questions, 4)
        num_answers = numpy.asarray([numpy.count_nonzero(ta) for ta in test_answers])
        assert numpy.all(num_answers == 1), "Some questions do not have exactly one answer"
        test_labels = numpy.argmax(test_answers, axis=1)

        # Indexing test data
        print("Indexing test data", file=sys.stderr)
        test_indices = self.data_indexer.process_data(test_lines, for_train=False)
        if in_shift_reduce_format:
            test_transitions, test_elements = self.data_indexer.get_shift_reduce_sequences(test_indices)
            test_transitions = self.data_indexer.pad_indices(test_transitions, max_length=max_length)
            test_elements = self.data_indexer.pad_indices(test_elements, max_length=max_length)
            # Make int32 array so that Keras will view them as indices.
            test_elements = numpy.asarray(test_elements, dtype='int32')
            # Adding trailing dimension to match TreeLSTM's input
            test_transitions = numpy.expand_dims(test_transitions, axis=-1)
            # The following needs to be a list for Keras to process them as inputs
            test_input = [test_transitions, test_elements]
        else:
            test_indices = self.data_indexer.pad_indices(test_indices, max_length=max_length)
            # Make int32 array so that Keras will view them as indices.
            test_input = numpy.asarray(test_indices, dtype='int32')
        return test_labels, test_input

    def score(self, test_input):
        probabilities = self.model.predict(test_input)
        return probabilities[:, 1] # Return p(t|x) only

    def evaluate(self, labels, test_input):
        # labels: list(int). List of indices of correct answers
        # test_input: list(numpy arrays) or numpy array: input to score
        test_scores = self.score(test_input)
        num_questions = len(labels)
        # test_scores will be of length 4*num_questions. Let's reshape it
        # to a matrix of size (num_questions, 4) and take argmax of over
        # the four options for each question.
        test_predictions = numpy.argmax(test_scores.reshape(num_questions, 4), axis=1)
        num_correct = sum(test_predictions == labels)
        accuracy = float(num_correct)/num_questions
        return accuracy


class LSTMSolver(NNSolver):
    def __init__(self, model_prefix, **kwargs):
        super(LSTMSolver, self).__init__(model_prefix, **kwargs)

    def build_model(self, train_input, vocab_size):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        ## STEP 1: Initialze the input layer
        input_layer = Input(shape=train_input.shape[1:], dtype='int32')

        ## STEP 2: Embed the propositions by changing word indices to word vectors
        # Mask zero ensures that padding is not a parameter in the entire model.
        embed_layer = Embedding(input_dim=vocab_size, output_dim=self.embedding_size, mask_zero=True,
                                name='embedding')
        embed = embed_layer(input_layer)
        # Add a dropout to regularize the input representations
        regularized_embed = Dropout(0.5)(embed)

        ## STEP 3: Pass the sequences of word vectors through LSTM
        lstm_layer = LSTM(output_dim=self.embedding_size, W_regularizer=l2(0.01), U_regularizer=l2(0.01),
                          b_regularizer=l2(0.01), name='lstm')
        lstm_out = lstm_layer(regularized_embed)
        # Add a dropout after LSTM
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        ## STEP 4: Find p(true | proposition) by passing the outputs from LSTM through
        # an MLP with ReLU layers
        projection_layer = Dense(self.embedding_size/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        ## Step 5: Define crossentropy against labels as the loss and compile the model
        model = Model(input=input_layer, output=output_probabilities)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary(), file=sys.stderr)
        return model

    def set_max_sentence_length_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]

    def get_training_data(self):
        dataset = self.get_simple_training_input()
        dataset.pad_instances(self.max_sentence_length)
        self.max_sentence_length = dataset.max_length()
        return dataset.as_training_data()

    def get_validation_data(self):
        dataset = Dataset.read_from_file(self.validation_file)
        self.data_indexer.index_dataset(dataset)


class TreeLSTMSolver(NNSolver):
    def __init__(self, model_prefix, **kwargs):
        super(TreeLSTMSolver, self).__init__(model_prefix, **kwargs)

    def build_model(self, train_input, vocab_size):
        '''
        train_input: List of two numpy arrays: transitions and initial buffer
        '''
        ## STEP 1: Initialze the two inputs
        transitions, _ = train_input
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
        embed_layer = Embedding(input_dim=vocab_size, output_dim=self.embedding_size, mask_zero=True,
                                name='embedding')
        embed = embed_layer(buffer_input)
        # Add a dropout to regularize the input representations
        regularized_embed = Dropout(0.5)(embed)

        ## STEP 3: Merge transitions and buffer (TreeLSTM only)
        lstm_input = merge([transitions_input, regularized_embed], mode='concat')

        ## STEP 4: Pass the sequences of word vectors through TreeLSTM
        lstm_layer = TreeCompositionLSTM(stack_limit, buffer_ops_limit,
                                         output_dim=self.embedding_size, W_regularizer=l2(0.01),
                                         U_regularizer=l2(0.01), V_regularizer=l2(0.01),
                                         b_regularizer=l2(0.01), name='treelstm')
        lstm_out = lstm_layer(lstm_input)
        regularized_lstm_out = Dropout(0.2)(lstm_out)

        ## STEP 5: Find p(true | proposition) by passing the encoded proposition through
        # MLP with ReLU followed by softmax
        projection_layer = Dense(self.embedding_size/2, activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_lstm_out))

        ## Step 6: Define crossentropy against labels as the loss and compile the model
        model = Model(input=[transitions_input, buffer_input], output=output_probabilities)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(model.summary(), file=sys.stderr)
        return model

    def set_max_sentence_length_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]

    def get_training_data(self):
        dataset = self.get_simple_training_input()
        dataset.pad_instances()
        train_input, labels = dataset.as_training_data()

        transitions, elements = self.data_indexer.get_shift_reduce_sequences(train_input)

        # TODO(matt): we need to pad the transitions / element sequences to the same length.  We
        # can probably remove the dataset.pad_instances() call above, and do a different kind of
        # padding here.  As it is, this code probably does not work.
        self.max_sentence_length = max(len(t) for t in transitions)

        # Make int32 array so that Keras will view them as indices.
        elements = numpy.asarray(elements, dtype='int32')

        # TreeLSTM's transitions input has an extra trailing dimension for
        # concatenation. This has to match.
        transitions = numpy.expand_dims(transitions, axis=-1)

        return (transitions, elements), labels


def main():
    argparser = argparse.ArgumentParser(description="Simple proposition scorer")
    argparser.add_argument('--positive_train_file', type=str)
    argparser.add_argument('--negative_train_file', type=str)
    argparser.add_argument('--validation_file', type=str)
    argparser.add_argument('--test_file', type=str)
    argparser.add_argument('--use_tree_lstm', help="Use TreeLSTM composition", action='store_true')
    argparser.add_argument('--length_upper_limit', type=int,
                           help="Upper limit on length of training data. Ignored during testing.")
    argparser.add_argument('--max_train_size', type=int,
                           help="Upper limit on the size of training data")
    argparser.add_argument('--num_epochs', type=int, default=20,
                           help="Number of train epochs (20 by default)")
    argparser.add_argument('--use_model_from_epoch', type=int,
                           help="Use the model from a particular epoch (use the best saved model if empty)")
    argparser.add_argument("--model_serialization_prefix",
                           help="Prefix for saving and loading model files")
    args = argparser.parse_args()
    solver_class = TreeLSTMSolver if args.use_tree_lstm else LSTMSolver
    nn_solver = solver_class(args.model_serialization_prefix,
                             max_sentence_length=args.length_upper_limit)

    if not args.positive_train_file:
        # Training file is not given. There must be a serialized model.
        print("Loading scoring model from disk", file=sys.stderr)
        custom_objects = {"TreeCompositionLSTM": TreeCompositionLSTM}
        nn_solver.load_model(args.use_model_from_epoch, custom_objects)
        # input shape of scoring model is (samples, max_length)
    else:
        assert args.validation_file is not None, "Validation data is needed for training"
        print("Reading training data", file=sys.stderr)
        positive_lines = [x.strip() for x in codecs.open(args.positive_train_file, "r", "utf-8").readlines()]
        if args.negative_train_file:
            negative_lines = [x.strip() for x in codecs.open(args.negative_train_file, "r", "utf-8").readlines()]
        else:
            negative_lines = []
        (inputs, labels) = nn_solver.prepare_training_data(
                positive_lines, negative_lines,
                max_length=args.length_upper_limit,
                in_shift_reduce_format=args.use_tree_lstm)

        train_sequence_length = inputs[0].shape[1] if args.use_tree_lstm  else inputs.shape[1]
        print("Reading validation data", file=sys.stderr)
        validation_lines = [x.strip() for x in codecs.open(args.validation_file, 'r', 'utf-8').readlines()]
        validation_labels, validation_input = nn_solver.prepare_test_data(
                validation_lines, max_length=train_sequence_length,
                in_shift_reduce_format=args.use_tree_lstm)
        if args.max_train_size is not None:
            print("Limiting training size to %d" % (args.max_train_size), file=sys.stderr)
            inputs = [input_[:args.max_train_size] for input_ in inputs] \
                    if args.use_tree_lstm else inputs[:args.max_train_size]
            labels = labels[:args.max_train_size]
        print("Training model", file=sys.stderr)
        nn_solver.train(inputs, labels, validation_input, validation_labels,
                        args.model_serialization_prefix, num_epochs=args.num_epochs)


    if args.test_file is not None:
        test_lines = [x.strip() for x in codecs.open(args.test_file, 'r', 'utf-8').readlines()]
        test_labels, test_input = nn_solver.prepare_test_data(
                test_lines, max_length=max_length, in_shift_reduce_format=args.use_tree_lstm)
        print("Scoring test data", file=sys.stderr)
        test_scores = nn_solver.score(test_input)
        accuracy = nn_solver.evaluate(test_labels, test_input)
        print("Test accuracy: %.4f" % accuracy, file=sys.stderr)
        assert len(test_scores) == len(test_lines)

        outfile = codecs.open("out.txt", "w", "utf-8")
        for score, line in zip(test_scores, test_lines):
            print(score, line, file=outfile)
        outfile.close()


if __name__ == "__main__":
    main()
