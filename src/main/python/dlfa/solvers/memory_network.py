import numpy

from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Dropout, merge
from keras.models import Model

from ..data.dataset import Dataset, IndexedDataset, TextDataset  # pylint: disable=unused-import
from ..layers.knowledge_selectors import selectors, DotProductKnowledgeSelector, ParameterizedKnowledgeSelector
from ..layers.memory_updaters import updaters
from ..layers.entailment_models import entailment_models
from .nn_solver import NNSolver


class MemoryNetworkSolver(NNSolver):
    '''
    We call this a Memory Network Solver because it has an attention over background knowledge, or
    "memory", similar to a memory network.  This implementation generalizes the architecture of the
    original memory network, though, and can be used to implement several papers in the literature,
    as well as some models that we came up with.

    Our basic architecture is as follows:
        Input: a sentence encoding and a set of background knowledge ("memory") encodings

        current_memory = sentence_encoding
        For each memory layer:
           attention_weights = knowledge_selector(current_memory, background)
           aggregated_background = weighted_sum(attention_weights, background)
           current_memory = memory_updater(current_memory, aggregated_background)
        final_score = entailment_model(aggregated_background, current_memory, sentence_encoding)

    There are thus three main knobs that can be turned (in addition to the number of memory
    layers):
        1. the knowledge_selector
        2. the memory_updater
        3. the entailment_model

    The original memory networks paper used the following:
        1. dot product (our DotProductKnowledgeSelector)
        2. sum
        3. linear classifier on top of current_memory

    The attentive reader in "Teaching Machines to Read and Comprehend", Hermann et al., 2015, used
    the following:
        1. a dense layer with a dot product bias (our ParameterizedKnowledgeSelector)
        2. Dense(K.concat([current_memory, aggregated_background]))
        3. Dense(current_memory)

    Our thought is that we should treat the last step as an entailment problem - does the
    background knowledge entail the input sentence?  Previous work was solving a different problem,
    so they used simpler models "entailment".
    '''


    def __init__(self, **kwargs):
        super(MemoryNetworkSolver, self).__init__(**kwargs)
        self.train_background = kwargs['train_background']
        self.positive_train_background = kwargs['positive_train_background']
        self.negative_train_background = kwargs['negative_train_background']
        self.validation_background = kwargs['validation_background']
        self.test_background = kwargs['test_background']
        self.debug_background = kwargs['debug_background']

        self.knowledge_selector = selectors[kwargs['knowledge_selector']]
        self.memory_updater = updaters[kwargs['memory_updater']]
        self.entailment_model = entailment_models[kwargs['entailment_model']](
                kwargs['entailment_num_hidden_layers'],
                kwargs['entailment_hidden_layer_width'],
                kwargs['entailment_hidden_layer_activation']
                )
        self.num_memory_layers = kwargs['num_memory_layers']

        self.max_knowledge_length = None

    @classmethod
    @overrides
    def update_arg_parser(cls, parser):
        super(MemoryNetworkSolver, cls).update_arg_parser(parser)

        parser.add_argument('--train_background', type=str)
        parser.add_argument('--positive_train_background', type=str)
        parser.add_argument('--negative_train_background', type=str)
        parser.add_argument('--validation_background', type=str)
        parser.add_argument('--test_background', type=str)
        parser.add_argument('--debug_background', type=str)

        parser.add_argument('--knowledge_selector', type=str, default='parameterized',
                            choices=selectors.keys(),
                            help='The kind of knowledge selector to use.  See '
                            'knowledge_selectors.py for details.')
        parser.add_argument('--memory_updater', type=str, default='dense_concat',
                            choices=updaters.keys(),
                            help='The kind of memory updaters to use.  See memory_updaters.py '
                            'for details.')
        parser.add_argument('--entailment_model', type=str, default='dense_memory_only',
                            choices=entailment_models.keys(),
                            help='The kind of entailment model to use.  See entailment_models.py '
                            'for details.')
        # TODO(matt): I wish there were a better way to do this...  You really want the entailment
        # model object to specify these arguments, and deal with them, instead of having NNSolver
        # have to know about them...  Not sure how to solve this.
        parser.add_argument('--entailment_num_hidden_layers', type=int, default=1,
                            help='Number of hidden layers in the entailment model')
        parser.add_argument('--entailment_hidden_layer_width', type=int, default=50,
                            help='Width of hidden layers in the entailment model')
        parser.add_argument('--entailment_hidden_layer_activation', type=int, default='relu',
                            help='Activation function for hidden layers in the entailment model')
        parser.add_argument('--num_memory_layers', type=int, default=1,
                            help="Number of memory layers in the network. (default 1)")

    @overrides
    def can_train(self) -> bool:
        has_train_background = (self.train_background is not None) or (
                self.positive_train_background is not None and
                self.negative_train_background is not None)
        has_validation_background = self.validation_background is not None
        has_background = has_train_background and has_validation_background
        return has_background and super(MemoryNetworkSolver, self).can_train()

    @overrides
    def can_test(self) -> bool:
        return self.test_background is not None and super(MemoryNetworkSolver, self).can_test()

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(MemoryNetworkSolver, cls)._get_custom_objects()
        custom_objects['DotProductKnowledgeSelector'] = DotProductKnowledgeSelector
        custom_objects['ParameterizedKnowledgeSelector'] = ParameterizedKnowledgeSelector
        return custom_objects

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]
        # TODO(matt): set the background length too, or does it matter?  Maybe the model doesn't
        # actually care?

    @overrides
    def _build_model(self):
        # Steps 1 and 2: Convert inputs to sequences of word vectors, for both the proposition
        # inputs and the knowledge inputs.
        proposition_input_layer, proposition_embedding = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,))
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
                input_shape=(self.max_knowledge_length, self.max_sentence_length),
                is_time_distributed=True)

        # Step 3: Encode the two embedded inputs using the same encoder.  We could replace the LSTM
        # below with fancier encoders depending on the input.
        proposition_encoder = self._get_sentence_encoder()

        # Knowledge encoder will have the same encoder running on a higher order tensor.
        # i.e., proposition_encoder: (samples, num_words, word_dim) -> (samples, word_dim)
        # and knowledge_encoder: (samples, knowledge_len, num_words, word_dim) ->
        #                       (samples, knowledge_len, word_dim)
        # TimeDistributed generally loops over the second dimension.
        knowledge_encoder = TimeDistributed(proposition_encoder, name='knowledge_encoder')
        encoded_proposition = proposition_encoder(proposition_embedding)  # (samples, word_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)  # (samples, knowledge_len, word_dim)

        # Step 4: Merge the two encoded representations and pass into the knowledge backed scorer.
        # At each step in the following loop, we take the proposition encoding, or the output of
        # the previous memory layer, merge it with the knowledge encoding and pass it to the
        # current memory layer.
        current_memory = encoded_proposition
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
            merged_encoded_rep = merge([current_memory, encoded_knowledge],
                                       mode=merge_mode,
                                       output_shape=merged_shape)

            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_selector = self.knowledge_selector(name='knowledge_selector_%d' % i)
            attention_weights = knowledge_selector(regularized_merged_rep)
            # Defining weighted average as a custom merge mode. Takes two inputs: data and weights
            # ndim of weights is one less than data.
            weighted_average = lambda avg_inputs: K.sum(avg_inputs[0] * K.expand_dims(avg_inputs[1], dim=-1),
                                                        axis=1)
            # input shapes: (samples, knowledge_len, input_dim), (samples, knowledge_len)
            # output shape: (samples, input_dim)
            weighted_average_shape = lambda input_shapes: (input_shapes[0][0], input_shapes[0][2])
            attended_knowledge = merge([encoded_knowledge, attention_weights],
                                       mode=weighted_average,
                                       output_shape=weighted_average_shape)
            memory_updater = self.memory_updater(output_dim=self.embedding_size,
                                                 name='memory_updater_%d' % i)
            current_memory = memory_updater.update(current_memory, attended_knowledge)

        # Step 5: Finally, run the sentence encoding, the current memory, and the attended
        # background knowledge through an entailment model to get a final true/false score.
        entailment_output = self.entailment_model.classify(encoded_proposition,
                                                           current_memory,
                                                           attended_knowledge)

        # Step 6: Define the model, and return it. The model will be compiled and trained by the
        # calling method.
        memory_network = Model(input=[proposition_input_layer, knowledge_input_layer],
                               output=entailment_output)
        return memory_network

    @overrides
    def _get_training_data(self):
        if self.train_file:
            dataset = TextDataset.read_from_file(self.train_file)
            background_dataset = TextDataset.read_background_from_file(dataset,
                                                                       self.train_background)
        else:
            positive_dataset = TextDataset.read_from_file(self.positive_train_file, label=True)
            positive_background = TextDataset.read_background_from_file(positive_dataset,
                                                                        self.positive_train_background)
            negative_dataset = TextDataset.read_from_file(self.negative_train_file, label=False)
            negative_background = TextDataset.read_background_from_file(negative_dataset,
                                                                        self.negative_train_background)
            background_dataset = positive_background.merge(negative_background)
        if self.max_training_instances is not None:
            background_dataset = background_dataset.truncate(self.max_training_instances)
        self.data_indexer.fit_word_dictionary(background_dataset)
        self.training_dataset = background_dataset
        return self.prep_labeled_data(background_dataset, for_train=True)

    @overrides
    def _get_validation_data(self):
        dataset = TextDataset.read_from_file(self.validation_file)
        background_dataset = TextDataset.read_background_from_file(dataset, self.validation_background)
        self.validation_dataset = background_dataset
        return self._prep_question_dataset(background_dataset)

    @overrides
    def _get_test_data(self):
        dataset = TextDataset.read_from_file(self.test_file)
        background_dataset = TextDataset.read_background_from_file(dataset, self.test_background)
        return self._prep_question_dataset(background_dataset)

    @overrides
    def _get_debug_dataset_and_input(self):
        dataset = TextDataset.read_from_file(self.debug_file)
        background_dataset = TextDataset.read_background_from_file(dataset, self.debug_background)
        # Now get inputs, and ignore the labels (background_dataset has them)
        inputs, _ = self.prep_labeled_data(background_dataset, for_train=False)
        return background_dataset, inputs

    @overrides
    def prep_labeled_data(self, dataset: Dataset, for_train: bool):
        """
        Not much to do here, as IndexedBackgroundInstance does most of the work.
        """
        max_lengths = [self.max_sentence_length, self.max_knowledge_length]
        processed_dataset = self._index_and_pad_dataset(dataset, max_lengths)
        if for_train:
            max_lengths = processed_dataset.max_lengths()
            self.max_sentence_length = max_lengths[0]
            self.max_knowledge_length = max_lengths[1]
        inputs, labels = processed_dataset.as_training_data()
        sentences, background = zip(*inputs)
        sentences = numpy.asarray(sentences)
        background = numpy.asarray(background)
        return [sentences, background], numpy.asarray(labels)

    def get_debug_layer_names(self):
        debug_layer_names = []
        for layer in self.model.layers:
            if "knowledge_selector" in layer.name:
                debug_layer_names.append(layer.name)
        return debug_layer_names

    def debug(self, debug_dataset, debug_inputs, epoch: int):
        """
        A debug_model must be defined by now. Run it on debug data and print the
        appropriate information to the debug output.
        """
        debug_output_file = open("%s_debug_%d.txt" % (self.model_prefix, epoch), "w")
        scores = self.score(debug_inputs)
        attention_outputs = self.debug_model.predict(debug_inputs)
        if self.num_memory_layers == 1:
            attention_outputs = [attention_outputs]
        # Collect values from all hops of attention for a given instance into attention_values.
        for instance, score, *attention_values in zip(debug_dataset.instances,
                                                      scores, *attention_outputs):
            sentence = instance.text
            background_info = instance.background
            label = instance.label
            positive_score = score[1]  # Only get p(t|x)
            # Remove the attention values for padding
            attention_values = [values[-len(background_info):] for values in attention_values]
            print("Sentence: %s" % sentence, file=debug_output_file)
            print("Label: %s" % label, file=debug_output_file)
            print("Assigned score: %.4f" % positive_score, file=debug_output_file)
            print("Weights on background:", file=debug_output_file)
            for i, background_i in enumerate(background_info):
                if i >= len(attention_values[0]):
                    # This happens when IndexedBackgroundInstance.pad() ignored some
                    # sentences (at the end). Let's ignore them too.
                    break
                all_hops_attention_i = ["%.4f" % values[i] for values in attention_values]
                print("\t%s\t%s" % (" ".join(all_hops_attention_i), background_i),
                      file=debug_output_file)
            print(file=debug_output_file)
        debug_output_file.close()
