from __future__ import print_function
import sys
import argparse
import codecs
from collections import defaultdict
import random
import numpy

from keras import backend as K
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dropout, merge
from keras.regularizers import l2
from keras.models import Model

from index_data import DataIndexer
from knowledge_backed_scorers import KnowledgeBackedDense
from nn_solver import NNSolver

class MemoryNetworkSolver(NNSolver):
    def index_inputs(self, inputs, for_train=True, length_cutoff=None, one_per_line=True):
        '''
        inputs: list((id, line)): List of index, input tuples.
            id is a identifier for each input to help link input sentences
            with corresponding knowledge
            line can either be a sentence or a logical form, part of
            either the propositions or the background knowledge.
        for_train: We want to update the word index only if we are processing
            training data. This flag will be passed to DataIndexer's process_data.
        length_cutoff: If not None, the inputs greater than this length will be 
            ignored. To keep the inputs aligned with ids, this will not be passed
            to DataIndexer's process_data, but instead used to postprocess the indices.
        one_per_line (bool): If set, this means there is only one data element per line.
            If not, it is assumed that there are multiple tab separated elements, all
            corresponding to the same id.

        returns: dict{id: list(indices)}: A mapping from id to a list of indices, each
            of which is a list of indices of words in a sentence.
            If the input corresponds to input sentences, each outer list contains only
            one element. But if it is background information, the list will most likely
            have multiple elements.
        '''
        if one_per_line:
            input_ids, input_lines = zip(*inputs)
        else:
            input_ids = []
            input_lines = []
            for input_id, input_line in inputs:
                for input_element in input_line.split("\t"):
                    input_ids.append(input_id)
                    input_lines.append(input_element)
        # TODO: process_data is going to change soon, and will no longer return
        # the number of propositions. Change this then.
        # Data indexer also returns number of propositions per line. Ignore it.
        _, indexed_input_lines = self.data_indexer.process_data(input_lines, 
                separate_propositions=False, for_train=for_train)
        mapped_indices = defaultdict(list)
        for input_id, indexed_input_line in zip(input_ids, indexed_input_lines):
            input_length = len(indexed_input_line)
            if length_cutoff is not None:
                if input_length <= length_cutoff:
                    mapped_indices[input_id].append(indexed_input_line)
            else:
                mapped_indices[input_id].append(indexed_input_line)
        return mapped_indices

    def train(self, proposition_inputs, knowledge_inputs, labels, validation_input,
            validation_labels, embedding_size=50, num_memory_layers=1, num_epochs=20,
            vocab_size=None):
        '''
        proposition_inputs: numpy_array(samples, num_words; int32): Indices of words
            in labeled propositions
        knowledge_inputs: numpy_array(samples, knowledge_len, num_words; int32): Indices
            of words in background facts that correspond to the propositions.
        labels: numpy_array(samples, 2): One-hot vectors indicating true/false
        validation_input: List containing processed proposition and knowledge inputs
            of validation data used for early stopping
        validation_labels: Similar to training labels, but for validation.
        num_memory_layers: Number of KnowledgeBackedDenseLayers to use for scoring.
        '''
        if not vocab_size:
            vocab_size = self.data_indexer.get_vocab_size()

        ## Step 1: Define the two inputs (propositions and knowledge)
        proposition_input = Input(shape=(proposition_inputs.shape[1:]), dtype='int32')
        knowledge_input = Input(shape=(knowledge_inputs.shape[1:]), dtype='int32')

        ## Step 2: Embed the two inputs using the same embedding matrix and apply dropout
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        # We need a timedistributed variant of the embedding (with same weights) to pass
        # the knowledge tensor in, and get a 4D tensor out.
        time_distributed_embedding = TimeDistributed(embedding)
        proposition_embed = embedding(proposition_input)  # (samples, num_words, word_dim)
        knowledge_embed = time_distributed_embedding(knowledge_input)  # (samples, knowledge_len, num_words, word_dim)
        regularized_proposition_embed = Dropout(0.5)(proposition_embed)
        regularized_knowledge_embed = Dropout(0.5)(knowledge_embed)

        ## Step 3: Encode the two embedded inputs using the same encoder
        # Can replace the LSTM below with fancier encoders depending on the input.
        proposition_encoder = LSTM(output_dim=embedding_size, W_regularizer=l2(0.01), 
                U_regularizer=l2(0.01), b_regularizer=l2(0.01), name='encoder')
        # Knowledge encoder will have the same encoder running on a higher order tensor.
        # i.e., proposition_encoder: (samples, num_words, word_dim) -> (samples, word_dim)
        # and knowledge_encoder: (samples, knowledge_len, num_words, word_dim) -> 
        #                       (samples, knowledge_len, word_dim)
        # TimeDistributed generally loops over the second dimension.
        knowledge_encoder = TimeDistributed(proposition_encoder, name='knowledge_encoder')
        encoded_proposition = proposition_encoder(regularized_proposition_embed)  # (samples, word_dim)
        encoded_knowledge = knowledge_encoder(regularized_knowledge_embed)  # (samples, knowledge_len, word_dim)

        ## Step 4: Merge the two encoded representations and pass into the knowledge backed 
        # scorer
        # At each step in the following loop, we take the proposition encoding,
        # or the output of the previous memory layer, merge it with the knowledge
        # encoding and pass it to the current memory layer (KnowledgeBackedDense).
        next_memory_layer_input = encoded_proposition
        for i in range(num_memory_layers):
            # We want to merge a matrix and a tensor such that the new tensor will have one
            # additional row (at the beginning) in all slices.
            # (samples, word_dim) + (samples, knowledge_len, word_dim) 
            #       -> (samples, 1 + knowledge_len, word_dim)
            # Since this is an unconventional merge, define a customized lambda merge.
            # Keras cannot infer the shape of the output of a lambda function, so make
            # that explicit.
            merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], 
                        dim=1), layer_outs[1]], axis=1)
            merged_shape = lambda layer_out_shapes: (layer_out_shapes[1][0],
                    layer_out_shapes[1][1] + 1, layer_out_shapes[1][2])
            merged_encoded_rep = merge([next_memory_layer_input, encoded_knowledge], 
                    mode=merge_mode, output_shape=merged_shape)
            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_backed_projector = KnowledgeBackedDense(output_dim=embedding_size,
                    name='memory_layer_%d' % i)
            memory_layer_output = knowledge_backed_projector(merged_encoded_rep)
            next_memory_layer_input = memory_layer_output

        ## Step 5: Finally score the projection.
        softmax = Dense(output_dim=2, activation='softmax', name='softmax')
        softmax_output = softmax(memory_layer_output)

        ## Step 6: Define the model, compile and train it.
        memory_network = Model(input=[proposition_input, knowledge_input], 
                output=softmax_output)
        memory_network.compile(loss='categorical_crossentropy', optimizer='adam')
        print(memory_network.summary(), file=sys.stderr)
        self.model = memory_network
        best_accuracy = 0.0
        for epoch_id in range(num_epochs):
            # History callback contains the losses of all training samples
            print("Epoch %d" % epoch_id, file=sys.stderr)
            memory_network.fit([proposition_inputs, knowledge_inputs], labels, nb_epoch=1)
            accuracy = self.evaluate(validation_labels, validation_input)
            print("Validation accuracy: %.4f" % accuracy, file=sys.stderr)
            if accuracy < best_accuracy:
                print("Stopping training", file=sys.stderr)
                break
            else:
                best_accuracy = accuracy
                self.save_model("memory_network_lstm", epoch_id)

    def prepare_data(self, proposition_lines, knowledge_lines, for_train=True):
        # Common data preparation function for both train and test data. 
        proposition_tuples = [x.split("\t") for x in proposition_lines]
        assert all([len(proposition_tuple) == 2 for proposition_tuple in 
            proposition_tuples]), "Malformed proposition input"
        # Keep track of maximum sentence length and number of sentences for padding
        # There are two kinds of knowledge padding coming up:
        # length padding: to make all background sentences the same length, done using 
        #   data indexer's pad_indices function in functions specific for train and test.
        # num padding: to make the number of background sentences the same for all 
        #   propositions, done by adding required number of sentences with just padding
        #   in this function itself.
        max_num_knowledge = 0  # for num padding
        # Separate all background knowledge corresponding to a sentence into multiple 
        # elements in the list having the same id.
        knowledge_tuples = []
        for line in knowledge_lines:
            parts = line.split("\t")
            # First part is the sentence index. Ignore it.
            num_knowledge = len(parts) - 1
            # Ignore the line if it is blank.
            if num_knowledge < 1:
                continue
            if num_knowledge > max_num_knowledge:
                max_num_knowledge = num_knowledge
            knowledge_tuples.append((parts[0], "\t".join(parts[1:])))
        mapped_proposition_indices = self.index_inputs(proposition_tuples, for_train=for_train)
        mapped_knowledge_indices = self.index_inputs(knowledge_tuples, for_train=for_train,
                one_per_line=False)
        proposition_inputs = []
        knowledge_inputs = []
        for proposition_id, proposition_indices in mapped_proposition_indices.items():
            # Proposition indices is a list of list of indices, but since there is only
            # one proposition for each index, just take the first (and only) list from 
            # the outer list.
            proposition_inputs.append(proposition_indices[0])
            knowledge_input = mapped_knowledge_indices[proposition_id]
            num_knowledge = len(knowledge_input)
            if num_knowledge < max_num_knowledge:
                # Num padding happening here. Since we will do length padding later,
                # we will add padding of length 1 each.
                for _ in range(max_num_knowledge - num_knowledge):
                    knowledge_input = [[0]] + knowledge_input
            knowledge_inputs.append(knowledge_input)
            # knowledge and proposition inputs are not length padded yet.
        return proposition_inputs, knowledge_inputs

    def prepare_training_data(self, positive_proposition_lines, positive_knowledge_lines, 
            negative_proposition_lines, negative_knowledge_lines, max_length=None):
        positive_proposition_inputs, positive_knowledge_inputs = self.prepare_data(
                positive_proposition_lines, positive_knowledge_lines, for_train=True)
        negative_proposition_inputs, negative_knowledge_inputs = self.prepare_data(
                negative_proposition_lines, negative_knowledge_lines, for_train=True)
        proposition_inputs = positive_proposition_inputs + negative_proposition_inputs
        knowledge_inputs = positive_knowledge_inputs + negative_knowledge_inputs
        num_positive_inputs = len(positive_proposition_inputs)
        num_negative_inputs = len(negative_proposition_inputs)
        # We're almost done. We just need to pad the inputs to be able to make them
        # numpy arrays.
        if not max_length:
            # Determine maximum sentence length in propositions and knowledge. We need
            # this to make sure all are length-padded to the same size. First find out max
            # proposition length, and then do the same for knowledge.
            # proposition_inputs are of shape (num_samples, num_words)
            max_proposition_length = max([len(proposition) for proposition in 
                proposition_inputs])
            max_knowledge_length = 0
            # knowledge_inputs are of shape (num_samples, num_sentences, num_words)
            for knowledge_index_list in knowledge_inputs:
                max_knowledge_length = max(max_knowledge_length, 
                        max([len(indices) for indices in knowledge_index_list]))
            max_length = max(max_proposition_length, max_knowledge_length)
        # Length padding proposition indices:
        proposition_inputs = self.data_indexer.pad_indices(proposition_inputs, max_length)
        # Length padding knowledge indices
        knowledge_inputs = [self.data_indexer.pad_indices(knowledge_input, max_length) 
                for knowledge_input in knowledge_inputs]
        # one hot labels: [1, 0] for positive, [0, 1] for negative
        labels = [[1,0] for _ in range(num_positive_inputs)] + \
                [[0,1] for _ in range(num_negative_inputs)]
        # Shuffle propositions, knowledge and labels in unison.
        all_inputs = zip(proposition_inputs, knowledge_inputs, labels)
        random.shuffle(all_inputs)
        proposition_inputs, knowledge_inputs, labels = zip(*all_inputs)
        # Make numpy arrays and return these.
        proposition_inputs = numpy.asarray(proposition_inputs)
        knowledge_inputs = numpy.asarray(knowledge_inputs)
        labels = numpy.asarray(labels)
        # Keras expects inputs as list. So returning it in a format where we can
        # directly use the inputs.
        return labels, [proposition_inputs, knowledge_inputs]

    def prepare_test_data(self, labeled_proposition_lines, knowledge_lines, max_length):
        '''
        proposition_lines: list(str): List of tab-separated strings, with first column 
            being sentence index (for knowledge mapping), second column indicating
            true/false and the third column the sentence
        knowledge_lines: list(str): List of tab-separated strings, first column sentence
            index (for knowledge mapping), and second column the sentence
        '''
        proposition_line_parts = [line.split("\t") for line in labeled_proposition_lines]
        # Make sure that every line has three parts
        assert all([len(parts) == 3 for parts in proposition_line_parts])
        test_labels = [int(parts[1]) for parts in proposition_line_parts]
        # Putting this in the two column format that process_data expects.
        test_proposition_lines = ["\t".join([parts[0], parts[2]]) for parts in
                proposition_line_parts]
        proposition_inputs, knowledge_inputs = self.prepare_data(test_proposition_lines,
                knowledge_lines, for_train=False)
        # Length padding proposition indices:
        proposition_inputs = self.data_indexer.pad_indices(proposition_inputs, max_length)
        # Length padding knowledge indices
        knowledge_inputs = [self.data_indexer.pad_indices(knowledge_input, max_length) 
                for knowledge_input in knowledge_inputs]
        # We want to return the indices of correct answers since that is what the evaluate 
        # function in nn_solver expects.
        num_questions = len(test_labels)/4
        test_answers = numpy.asarray(test_labels).reshape(num_questions, 4)
        assert numpy.all(numpy.asarray([numpy.count_nonzero(ta) for ta in 
            test_answers]) == 1), "Some questions do not have exactly one answer"
        test_labels = numpy.argmax(test_answers, axis=1)
        proposition_inputs = numpy.asarray(proposition_inputs, dtype='int32')
        knowledge_inputs = numpy.asarray(knowledge_inputs, dtype='int32')
        # Keras expects inputs as list. So returning it in a format where we can
        # directly use the inputs. For example, in the evaluate function in nn_solver.
        return test_labels, [proposition_inputs, knowledge_inputs]
        
if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Memory network solver")
    argparser.add_argument('--positive_train_input', type=str)
    argparser.add_argument('--positive_train_background', type=str)
    argparser.add_argument('--negative_train_input', type=str)
    argparser.add_argument('--negative_train_background', type=str)
    argparser.add_argument('--validation_input', type=str)
    argparser.add_argument('--validation_background', type=str)
    argparser.add_argument('--test_input', type=str)
    argparser.add_argument('--test_background', type=str)
    argparser.add_argument('--num_memory_layers', type=int, 
            help="Number of memory layers in the network. (default 1)", default=1)
    argparser.add_argument('--length_upper_limit', type=int, 
            help="Upper limit on length of training data. Ignored during testing.")
    argparser.add_argument('--max_train_size', type=int, 
            help="Upper limit on the size of training data")
    argparser.add_argument('--num_epochs', type=int, default=20,
            help="Number of train epochs (20 by default)")
    argparser.add_argument('--use_model_from_epoch', type=int, default=0, 
            help="Use the model from a particular epoch (0 by default)")
    argparser.add_argument('--output_file', type=str, default="out.txt", 
            help="Name of the file to print the test output. out.txt by default")
    args = argparser.parse_args()
    nn_solver = MemoryNetworkSolver()

    if not args.positive_train_input:
        # Training file is not given. There must be a serialized model.
        print("Loading scoring model from disk", file=sys.stderr)
        model_name_prefix = "memory_network_lstm"
        custom_objects = {"KnowledgeBackedDense": KnowledgeBackedDense}
        nn_solver.load_model(model_name_prefix, args.use_model_from_epoch, 
                custom_objects=custom_objects)
    else:
        assert args.positive_train_background is not None, "Positive background data required for training"
        assert args.negative_train_input is not None, "Negative data required for training"
        assert args.negative_train_background is not None, "Negative background data required for training"
        assert args.validation_input is not None, "Validation data is needed for training"
        assert args.validation_background is not None, "Validation background is needed for training"
        print("Reading training data", file=sys.stderr)
        positive_proposition_lines = [x.strip() for x in codecs.open(
            args.positive_train_input, "r", "utf-8").readlines()]
        negative_proposition_lines = [x.strip() for x in codecs.open(
            args.negative_train_input, "r", "utf-8").readlines()]
        positive_knowledge_lines = [x.strip() for x in codecs.open(
            args.positive_train_background, "r", "utf-8").readlines()]
        negative_knowledge_lines = [x.strip() for x in codecs.open(
            args.negative_train_background, "r", "utf-8").readlines()]
        train_labels, train_inputs = nn_solver.prepare_training_data(
                positive_proposition_lines, positive_knowledge_lines,
                negative_proposition_lines, negative_knowledge_lines,
                max_length=args.length_upper_limit)
        if args.max_train_size is not None:
            print("Limiting training size to %d" % (args.max_train_size), file=sys.stderr)
            train_inputs = [input_[:args.max_train_size] for input_ in train_inputs]
            train_labels = train_labels[:args.max_train_size]
        train_proposition_inputs, train_knowledge_inputs = train_inputs
        # Record max_length to use it for processing validation data.
        train_sequence_length = train_proposition_inputs.shape[1]
        print("Reading validation data", file=sys.stderr)
        validation_proposition_lines = [x.strip() for x in codecs.open(
            args.validation_input,'r', 'utf-8').readlines()]
        validation_knowledge_lines = [x.strip() for x in codecs.open(
            args.validation_background,'r', 'utf-8').readlines()]
        validation_labels, validation_input = nn_solver.prepare_test_data(
                validation_proposition_lines, validation_knowledge_lines,
                train_sequence_length)
        print("Training model", file=sys.stderr)
        nn_solver.train(train_proposition_inputs, train_knowledge_inputs, train_labels,
                validation_input, validation_labels,
                num_memory_layers=args.num_memory_layers, num_epochs=args.num_epochs)
    
    # We need this for making sure that test sequences are not longer than what the trained model
    # expects.
    max_length = nn_solver.model.get_input_shape_at(0)[0][1]

    if args.test_input is not None:
        assert args.test_background is not None, "Test background file not provided"
        test_proposition_lines = [x.strip() for x in codecs.open(args.test_input,
            'r', 'utf-8').readlines()]
        test_knowledge_lines = [x.strip() for x in codecs.open(args.test_background,
            'r', 'utf-8').readlines()]
        test_labels, test_input = nn_solver.prepare_test_data(test_proposition_lines,
                test_knowledge_lines, max_length)
        print("Scoring test data", file=sys.stderr)
        test_scores = nn_solver.score(test_input)
        accuracy = nn_solver.evaluate(test_labels, test_input)
        print("Test accuracy: %.4f" % accuracy, file=sys.stderr)

        outfile = codecs.open(args.output_file, "w", "utf-8")
        for score, line in zip(test_scores, test_proposition_lines):
            print(score, line, file=outfile)
        outfile.close()

