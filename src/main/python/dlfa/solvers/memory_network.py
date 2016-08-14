from __future__ import print_function
import sys

import numpy

from keras import backend as K
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense, Dropout, merge
from keras.regularizers import l2
from keras.models import Model

from ..data.dataset import Dataset, IndexedDataset  # pylint: disable=unused-import
from ..layers.knowledge_backed_scorers import AttentiveReaderLayer, MemoryLayer
from .nn_solver import NNSolver

'''
TODO(pradeep): Replace memory layers in the following implementation, with a combination
of a KnowledgeSelector and the logic for updating memory given below.

Insight: Memory Networks and Attentive Readers have the following steps:
    0. Take knowledge input (z), query input (q)
    1. Knowledge Selection (returns knowledge weights)
    2. Knowledge Aggregation (returns weighted average)
    3. Memory Update (returns a memory representation to replace input)
    4. Optionally repeat 1-3 with output from 3 replacing q
    5. Pass aggregated knowledge from 2, and q to an entailment function.
Memory Network: Selector = SimpleKnowledgeSelector; Updater = merge layer with mode = sum
AttentiveReader: Selector = ParameterizedKnowledgeSelector; Updater = merge with mode = concat, followed by a dense layer.
'''

class MemoryNetworkSolver(NNSolver):

    memory_layer_choices = ['attentive', 'memory']

    def __init__(self, **kwargs):
        '''
        num_memory_layers: Number of KnowledgeBackedDenseLayers to use for scoring.
        '''
        super(MemoryNetworkSolver, self).__init__(**kwargs)
        if kwargs['memory_layer'] == 'attentive':
            self.memory_layer = AttentiveReaderLayer
        elif kwargs['memory_layer'] == 'memory':
            self.memory_layer = MemoryLayer
        else:
            raise RuntimeError("Unrecognized memory layer type: " + kwargs['memory_layer'])
        self.num_memory_layers = kwargs['num_memory_layers']

        self.train_background = kwargs['train_background']
        self.positive_train_background = kwargs['positive_train_background']
        self.negative_train_background = kwargs['negative_train_background']
        self.validation_background = kwargs['validation_background']
        self.test_background = kwargs['test_background']

        self.max_knowledge_length = None

    @classmethod
    def update_arg_parser(cls, parser):
        super(MemoryNetworkSolver, cls).update_arg_parser(parser)

        parser.add_argument('--train_background', type=str)
        parser.add_argument('--positive_train_background', type=str)
        parser.add_argument('--negative_train_background', type=str)
        parser.add_argument('--validation_background', type=str)
        parser.add_argument('--test_background', type=str)

        parser.add_argument('--memory_layer', type=str, default='attentive',
                            choices=cls.memory_layer_choices,
                            help='The kind of memory layer to use.  Options are "memory" and '
                            '"attentive".  See knowledge_backed_scorers.py for details.')
        parser.add_argument('--num_memory_layers', type=int, default=1,
                            help="Number of memory layers in the network. (default 1)")

    def can_train(self) -> bool:
        has_train_background = (self.train_background is not None) or (
                self.positive_train_background is not None and
                self.negative_train_background is not None)
        has_validation_background = self.validation_background is not None
        has_background = has_train_background and has_validation_background
        return has_background and super(MemoryNetworkSolver, self).can_train()

    def can_test(self) -> bool:
        return self.test_background is not None and super(MemoryNetworkSolver, self).can_test()

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(MemoryNetworkSolver, cls)._get_custom_objects()
        custom_objects['MemoryLayer'] = MemoryLayer
        custom_objects['AttentiveReaderLayer'] = AttentiveReaderLayer
        return custom_objects

    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]
        # TODO(matt): set the background length too, or does it matter?  Maybe the model doesn't
        # actually care?

    def _build_model(self, train_input):
        '''
        train_input: a tuple of (proposition_inputs, knowledge_inputs), each described below:
            proposition_inputs: numpy_array(samples, num_words; int32): Indices of words
                in labeled propositions
            knowledge_inputs: numpy_array(samples, knowledge_len, num_words; int32): Indices
                of words in background facts that correspond to the propositions.
        '''
        vocab_size = self.data_indexer.get_vocab_size()

        ## Step 1: Define the two inputs (propositions and knowledge)
        proposition_inputs, knowledge_inputs = train_input
        print("SHAPES:")
        print(proposition_inputs.shape, knowledge_inputs.shape)
        proposition_input = Input(shape=(proposition_inputs.shape[1:]), dtype='int32')
        knowledge_input = Input(shape=(knowledge_inputs.shape[1:]), dtype='int32')

        ## Step 2: Embed the two inputs using the same embedding matrix and apply dropout
        embedding = Embedding(input_dim=vocab_size, output_dim=self.embedding_size,
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
        proposition_encoder = LSTM(output_dim=self.embedding_size, W_regularizer=l2(0.01),
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
        # encoding and pass it to the current memory layer.
        next_memory_layer_input = encoded_proposition
        for i in range(self.num_memory_layers):
            # We want to merge a matrix and a tensor such that the new tensor will have one
            # additional row (at the beginning) in all slices.
            # (samples, word_dim) + (samples, knowledge_len, word_dim)
            #       -> (samples, 1 + knowledge_len, word_dim)
            # Since this is an unconventional merge, we define a customized lambda merge.
            # Keras cannot infer the shape of the output of a lambda function, so we make
            # that explicit.
            merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=1),
                                                           layer_outs[1]],
                                                          axis=1)
            merged_shape = lambda layer_out_shapes: \
                (layer_out_shapes[1][0], layer_out_shapes[1][1] + 1, layer_out_shapes[1][2])
            merged_encoded_rep = merge([next_memory_layer_input, encoded_knowledge],
                                       mode=merge_mode,
                                       output_shape=merged_shape)

            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_backed_projector = self.memory_layer(output_dim=self.embedding_size,
                                                           name='memory_layer_%d' % i)
            memory_layer_output = knowledge_backed_projector(regularized_merged_rep)
            next_memory_layer_input = memory_layer_output

        ## Step 5: Finally score the projection.
        softmax = Dense(output_dim=2, activation='softmax', name='softmax')
        softmax_output = softmax(memory_layer_output)

        ## Step 6: Define the model, compile and train it.
        memory_network = Model(input=[proposition_input, knowledge_input], output=softmax_output)
        memory_network.compile(loss='categorical_crossentropy', optimizer='adam')
        print(memory_network.summary(), file=sys.stderr)
        return memory_network

    def _get_training_data(self):
        if self.train_file:
            dataset = Dataset.read_from_file(self.train_file)
            background_dataset = Dataset.read_background_from_file(dataset, self.train_background)
        else:
            positive_dataset = Dataset.read_from_file(self.positive_train_file, label=True)
            positive_background = Dataset.read_background_from_file(positive_dataset,
                                                                    self.positive_train_background)
            negative_dataset = Dataset.read_from_file(self.negative_train_file, label=False)
            negative_background = Dataset.read_background_from_file(negative_dataset,
                                                                    self.negative_train_background)
            background_dataset = positive_background.merge(negative_background)
        if self.max_training_instances is not None:
            background_dataset = background_dataset.truncate(self.max_training_instances)
        self.data_indexer.fit_word_dictionary(background_dataset)
        return self.prep_labeled_data(background_dataset, for_train=True)

    def _get_validation_data(self):
        return self._read_question_data_with_background(self.validation_file, self.validation_background)

    def _get_test_data(self):
        return self._read_question_data_with_background(self.test_file, self.test_background)

    def _read_question_data_with_background(self, filename, background_filename):
        dataset = Dataset.read_from_file(filename)
        self._assert_dataset_is_questions(dataset)
        background_dataset = Dataset.read_background_from_file(dataset, background_filename)
        inputs, labels = self.prep_labeled_data(background_dataset, for_train=False)
        return inputs, self.group_by_question(labels)

    def prep_labeled_data(self, dataset: Dataset, for_train: bool):
        """
        Not much to do here, as IndexedBackgroundInstance does most of the work.
        """
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        indexed_dataset.pad_instances([self.max_sentence_length, self.max_knowledge_length])
        if for_train:
            max_lengths = indexed_dataset.max_lengths()
            self.max_sentence_length = max_lengths[0]
            self.max_knowledge_length = max_lengths[1]
        inputs, labels = indexed_dataset.as_training_data()
        sentences, background = zip(*inputs)
        sentences = numpy.asarray(sentences)
        background = numpy.asarray(background)
        return (sentences, background), numpy.asarray(labels)
