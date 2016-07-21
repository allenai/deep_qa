from collections import defaultdict

from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.models import Model

from index_data import DataIndexer
from knowledge_backed_scorers import KnowledgeBackedDense

class MemoryNetworkSolver(object):
    def __init__(self):
        self.data_indexer = DataIndexer()

    def index_inputs(self, inputs, for_train=True, max_length=None):
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
        '''
        input_ids, input_lines = zip(*inputs)
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
        embedding = Embedding(input_shape=vocab_size, output_dim=embedding_size, 
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
        # encoding and pass it to the next mem layer (KnowledgeBackedDense).
        next_memory_layer_input = encoded_proposition
        for i in range(num_memory_layers):
            # We want to merge a matrix and a tensor such that the new tensor will have one
            # additional row (at the beginning) in all slices.
            # (samples, word_dim) + (samples, knowledge_len, word_dim) 
            #       -> (samples, 1 + knowledge_len, word_dim)
            # Since this is an unconventional merge, define a customized lambda merge.
            merged_encoded_rep = merge([next_memory_layer_input, encoded_knowledge], 
                    mode=lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], 
                        dim=1), layer_outs[1]], axis=1))
            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_backed_projector = KnowledgeBackedDense(output_dim=embedding_size,
                    name='memory_layer_%d' % i)
            memory_layer_output = knowledge_backed_projector(merged_encoded_rep)
            next_memory_layer_input = memory_layer_output

        ## Step 5: Finally score the projection.
        softmax = Dense(output_dim=2, activation='softmax', name='softmax')
        softmax_output = softmax(memory_layer_output)

        memory_network = Model(input=[proposition_input, knowledge_input], 
                output=softmax_output)
        
