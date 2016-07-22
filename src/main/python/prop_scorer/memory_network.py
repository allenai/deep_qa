import sys
from collections import defaultdict

from keras import backend as K
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dropout, merge
from keras.regularizers import l2
from keras.models import Model

from index_data import DataIndexer
from knowledge_backed_scorers import KnowledgeBackedDense
from nn_solver import NNSolver

class MemoryNetworkSolver(NNSolver):
    def index_inputs(self, inputs, for_train=True, max_length=None, one_per_line=True):
        '''
        inputs: list((id, line)): List of index, input tuples.
            id is a identifier for each input to help link input sentences
            with corresponding knowledge
            line can either be a sentence or a logical form, part of
            either the propositions or the background knowledge
        for_train: We want to update the word index only if we are processing
            training data. This flag will be passed to DataIndexer's process_data
        max_length: If not None, the inputs greater than this length will be 
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
        indexed_input_lines = self.data_indexer.process_data(input_lines, 
                separate_propositions=False, for_train=for_train)
        assert len(input_ids) == len(indexed_input_lines)
        mapped_indices = defaultdict(list)
        for input_id, indexed_input_line in zip(input_ids, indexed_input_lines):
            if max_length is not None:
                if len(indexed_input_line) <= max_length:
                    mapped_indices[input_id].append(indexed_input_line)
            else:
                mapped_indices[input_id].append(indexed_input_line)
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
        positive_knowledge_tuples = []
        for line in positive_knowledge_lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            positive_knowledge_tuples.append((parts[0], "\t".join(parts[1:])))
        negative_knowledge_tuples = []
        for line in negative_knowledge_lines:
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            negative_knowledge_tuples.append((parts[0], "\t".join(parts[1:])))
        positive_proposition_indices = self.index_inputs(positive_proposition_tuples,
               for_train=True)
        negative_proposition_indices = self.index_inputs(negative_proposition_tuples,
               for_train=True)
        positive_knowledge_indices = self.index_inputs(positive_proposition_tuples,
               for_train=True, one_per_line=False)
        negative_knowledge_indices = self.index_inputs(negative_proposition_tuples,
               for_train=True, one_per_line=False)
        

    def prepare_test_data(self, data_lines, max_length=None):
