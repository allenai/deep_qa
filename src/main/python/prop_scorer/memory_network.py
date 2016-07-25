import sys
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
            either the propositions or the background knowledge
        for_train: We want to update the word index only if we are processing
            training data. This flag will be passed to DataIndexer's process_data
        length_cutoff: If not None, the inputs greater than this length will be 
            ignored. To keep the inputs aligned with ids, this will not be passed
            to DataIndexer's process_data, but instead used to postprocess the indices
        one_per_line (bool): If set, this means there is only one data element per line.
            If not, it is assumed that there are multiple tab separated elements, all
            corresponding to the same id.
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
        # Data indexer also reurns number of propositions per line. Ignore it.
        _, indexed_input_lines = self.data_indexer.process_data(input_lines, 
                separate_propositions=False, for_train=for_train)
        assert len(input_ids) == len(indexed_input_lines)
        mapped_indices = defaultdict(list)
        #max_length = 0
        for input_id, indexed_input_line in zip(input_ids, indexed_input_lines):
            input_length = len(indexed_input_line)
            #if input_length > max_length:
            #    max_length = input_length
            if length_cutoff is not None:
                if input_length <= length_cutoff:
                    mapped_indices[input_id].append(indexed_input_line)
            else:
                mapped_indices[input_id].append(indexed_input_line)
        #return max_length, mapped_indices
        return mapped_indices

    def train(self, proposition_indices, knowledge_indices, labels, embedding_size=50,
            num_memory_layers=1, num_epochs=20, vocab_size=None):
        '''
        proposition_indices: numpy_array(samples, num_words; int32): Indices of words
            in labeled propositions
        knowledge_indices: numpy_array(samples, knowledge_len, num_words; int32): Indices
            of words in background facts that correspond to the propositions.
        labels: numpy_array(samples, 2): One-hot vectors indicating true/false
        num_memory_layers: Number of KnowledgeBackedDenseLayers to use for scoring.
        '''
        # TODO: Also do validation, and use it for early stopping
        if not vocab_size:
            vocab_size = self.data_indexer.get_vocab_size()

        ## Step 1: Define the two inputs (propositions and knowledge)
        proposition_input = Input(shape=(proposition_indices.shape[1:]), dtype='int32')
        knowledge_input = Input(shape=(knowledge_indices.shape[1:]), dtype='int32')

        ## Step 2: Embed the two inputs using the same embedding matrix and apply dropout
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_size, 
                mask_zero=True, name='embedding')
        # We need a timedistributed variant of the embedding (with same weights) to pass
        # the knowledge tensor in, and get a 4D tensor out.
        time_distributed_embedding = TimeDistributed(embedding)
        proposition_embed = embedding(proposition_input) # (samples, num_words, word_dim)
        knowledge_embed = time_distributed_embedding(knowledge_input) # (samples, knowledge_len, num_words, word_dim)
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
        encoded_proposition = proposition_encoder(regularized_proposition_embed) # (samples, word_dim)
        encoded_knowledge = knowledge_encoder(regularized_knowledge_embed) # (samples, knowledge_len, word_dim)

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
        print >>sys.stderr, memory_network.summary()
        memory_network.fit([proposition_indices, knowledge_indices], labels)
        self.model = memory_network

    def prepare_training_data(self, positive_proposition_lines, positive_knowledge_lines, 
            negative_proposition_lines, negative_knowledge_lines, max_length=None):
        positive_proposition_tuples = [x.split("\t") for x in positive_proposition_lines]
        assert all([len(proposition_tuple) == 2 for proposition_tuple in 
            positive_proposition_tuples]), "Malformed positive proposition file"
        negative_proposition_tuples = [x.split("\t") for x in negative_proposition_lines]
        assert all([len(proposition_tuple) == 2 for proposition_tuple in 
            negative_proposition_tuples]), "Malformed negative proposition file"
        # Keep track of maximum sentence length and number of sentences for padding
        # There are two kinds of knowledge padding coming up:
        # length padding: to make all background sentences the same length, done using 
        #   data indexer's pad_indices function
        # num padding: to make the number of background sentences the same for all 
        #   propositions, done by adding required number of sentences with just padding
        #   in this function itself.
        max_knowledge_length = 0 # for length padding
        max_num_knowledge = 0 # for num padding
        # Separate all background knowledge corresponging to a sentence into multiple 
        # elements in the list having the same id.
        positive_knowledge_tuples = []
        for line in positive_knowledge_lines:
            parts = line.split("\t")
            # First part is the sentence index. Ignore it.
            num_knowledge = len(parts) - 1
            # Ignore the line if it is blank.
            if num_knowledge < 1:
                continue
            if num_knowledge > max_num_knowledge:
                max_num_knowledge = num_knowledge
            positive_knowledge_tuples.append((parts[0], "\t".join(parts[1:])))
        negative_knowledge_tuples = []
        for line in negative_knowledge_lines:
            parts = line.split("\t")
            num_knowledge = len(parts) - 1
            if num_knowledge < 1:
                continue
            if num_knowledge > max_num_knowledge:
                max_num_knowledge = num_knowledge
            negative_knowledge_tuples.append((parts[0], "\t".join(parts[1:])))

        positive_proposition_indices = self.index_inputs(positive_proposition_tuples,
                for_train=True)
        negative_proposition_indices = self.index_inputs(negative_proposition_tuples,
                for_train=True)
        positive_knowledge_indices = self.index_inputs(positive_knowledge_tuples,
                for_train=True, one_per_line=False)
        negative_knowledge_indices = self.index_inputs(negative_knowledge_tuples,
                for_train=True, one_per_line=False)
        if not max_length:
            # Find the max length of each knowledge sentence. We need this for length padding
            for word_indices in positive_knowledge_indices.values() + negative_knowledge_indices.values():
                knowledge_length = max([len(indices) for indices in word_indices])
                if knowledge_length > max_knowledge_length:
                    max_knowledge_length = knowledge_length
            # max_length is used for padding propositions. Make sure they are the same
            # length as knowledge sentences. This is because the same LSTM is used for
            # encoding both, and Keras expects the length of all sequences processed by
            # a Recurrent layer to be the same.
            max_length = max_knowledge_length
        else:
            max_knowledge_length = max_length
        proposition_inputs = []
        knowledge_inputs = []
        labels = []
        def _pad_knowledge(proposition_indices, knowledge_indices, label_one_hot):
            # label_one_hot: [0, 1] for positive, [1, 0] for negative.
            for proposition_id, proposition_indices in proposition_indices.items():
                # Proposition indices is a list of list of indices, but since there is only
                # one proposition for each index, just take the first (and only) list from 
                # the outer list.
                proposition_inputs.append(proposition_indices[0])
                knowledge_input = knowledge_indices[proposition_id]
                num_knowledge = len(knowledge_input)
                if num_knowledge < max_num_knowledge:
                    # Num padding happening here
                    for _ in range(max_num_knowledge - num_knowledge):
                        knowledge_input = [[0]] + knowledge_input 
                # Length padding happening here:
                padded_knowledge_input = self.data_indexer.pad_indices(knowledge_input,
                        max_length = max_knowledge_length)
                knowledge_inputs.append(padded_knowledge_input)
                labels.append(label_one_hot)
        # Now use the method above to pad both positive and negative knowledge indices
        # add them to the same knowledge_inputs array.
        _pad_knowledge(positive_proposition_indices, positive_knowledge_indices, [0,1])
        _pad_knowledge(negative_proposition_indices, negative_knowledge_indices, [1,0])
        # Shuffle the two inputs, and labels array in unison
        all_inputs = zip(proposition_inputs, knowledge_inputs, labels)
        random.shuffle(all_inputs)
        proposition_inputs, knowledge_inputs, labels = zip(*all_inputs)
        # Length padding proposition_inputs:
        padded_proposition_inputs = self.data_indexer.pad_indices(proposition_inputs,
                max_length=max_length)
        proposition_inputs = numpy.asarray(padded_proposition_inputs, dtype='int32')
        knowledge_inputs = numpy.asarray(knowledge_inputs, dtype='int32')
        labels = numpy.asarray(labels)
        return proposition_inputs, knowledge_inputs, labels

    def prepare_test_data(self, data_lines, max_length=None):
        # Take general parts from prepare_training data out, make is a separate function
        # and reuse it here.
        raise NotImplementedError
