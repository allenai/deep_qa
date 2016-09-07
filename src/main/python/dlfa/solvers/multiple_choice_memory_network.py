from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Dropout, merge
from keras.models import Model

from ..data.dataset import Dataset, IndexedDataset, TextDataset  # pylint: disable=unused-import
from .memory_network import MemoryNetworkSolver


class MultipleChoiceMemoryNetworkSolver(MemoryNetworkSolver):
    '''
    This is a MemoryNetworkSolver that is trained on multiple choice questions, instead of
    true/false questions.

    This needs changes to two areas of the code: (1) how the data is preprocessed, and (2) how the
    model is built.

    Instead of getting a list of positive and negative examples, we get a question with labeled
    answer options, only one of which can be true.  We then pass each option through the same
    basic structure as the MemoryNetworkSolver, and use a softmax at the end to get a final answer.

    This essentially just means making the MemoryNetworkSolver model time-distributed everywhere,
    and adding a final softmax.
    '''

    def __init__(self, **kwargs):
        super(MultipleChoiceMemoryNetworkSolver, self).__init__(**kwargs)
        self.num_options = 4  # TODO(matt): we'll handle the more general case later

    @overrides
    def can_train(self) -> bool:
        """
        Where a MemoryNetworkSolver allows separate positive and negative training files, we only
        allow a single train file, so we need to override this method.

        The train file must be a valid question file, as determined by
        Dataset.can_be_converted_to_questions(), but we don't check that here.
        """
        has_train = self.train_file is not None and self.train_background is not None
        has_validation = self.validation_file is not None and self.validation_background is not None
        return has_train and has_validation

    @overrides
    def _build_model(self):
        """
        This model is identical to the MemoryNetworkSolver model, except that everything is
        TimeDistributed once, to allow for num_options sentences and background per training
        instance.  We also have a final softmax at the end to select an answer option.

        TODO(matt): It'd be nice to not have to duplicate the memory network architecture here, but
        if it's possible to share more code, I'll figure out how to do that later.  It might be
        possible by overriding _get_sentence_encoder() to be TimeDistributed, and by some fancy
        private methods in MemoryNetworkSolver...
        """
        # Steps 1 and 2: Convert inputs to sequences of word vectors, for both the proposition
        # inputs and the knowledge inputs.
        proposition_input_layer, proposition_embedding = self._get_embedded_sentence_input(
                input_shape=(self.num_options, self.max_sentence_length,))
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
                input_shape=(self.num_options, self.max_knowledge_length, self.max_sentence_length))

        # Step 3: Encode the two embedded inputs using the sentence encoder.  Because we have
        # several sentences per instance, we need to TimeDistribute it first.
        proposition_encoder = TimeDistributed(self._get_sentence_encoder())

        # Knowledge encoder will have the same encoder running on a higher order tensor.
        knowledge_encoder = TimeDistributed(proposition_encoder, name='knowledge_encoder')

        # (batch_size, num_options, encoding_dim)
        encoded_proposition = proposition_encoder(proposition_embedding)

        # (batch_size, num_options, knowledge_len, encoding_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)

        # Step 4: Merge the two encoded representations and pass into the knowledge backed scorer.
        # At each step in the following loop, we take the proposition encoding, or the output of
        # the previous memory layer, merge it with the knowledge encoding and pass it to the
        # current memory layer.
        current_memory = encoded_proposition
        for i in range(self.num_memory_layers):
            # We want to merge a 3-mode tensor and a 4-mode tensor such that the new tensor will
            # have one additional row (at the beginning) in all slices.
            # (batch_size, num_options, encoding_dim) + (batch_size, num_options, knowledge_len, encoding_dim)
            #       -> (batch_size, num_options, 1 + knowledge_len, encoding_dim)
            # Since this is an unconventional merge, we define a customized lambda merge.
            # Keras cannot infer the shape of the output of a lambda function, so we make
            # that explicit.
            merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=2),
                                                           layer_outs[1]],
                                                          axis=2)
            merged_shape = lambda shapes: (shapes[1][0], shapes[1][1], shapes[1][2] + 1, shapes[1][3])
            merged_encoded_rep = merge([current_memory, encoded_knowledge],
                                       mode=merge_mode,
                                       output_shape=merged_shape)

            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_selector = TimeDistributed(self.knowledge_selector(),
                                                 name='knowledge_selector_%d' % i)
            attention_weights = knowledge_selector(regularized_merged_rep)
            # Defining weighted average as a custom merge mode. Takes two inputs: data and weights
            # ndim of weights is one less than data.
            weighted_average = lambda avg_inputs: K.sum(avg_inputs[0] * K.expand_dims(avg_inputs[1], dim=-1),
                                                        axis=2)
            # input shapes: (batch_size, num_options, knowledge_len, encoding_dim),
            #               (batch_size, num_options, knowledge_len)
            # output shape: (batch_size, num_options, encoding_dim)
            weighted_average_shape = lambda shapes: (shapes[0][0], shapes[0][1], shapes[0][3])
            attended_knowledge = merge([encoded_knowledge, attention_weights],
                                       mode=weighted_average,
                                       output_shape=weighted_average_shape)

            # To make this easier to TimeDistribute, we're going to concatenate the current memory
            # with the attended knowledge, and use that as the input to the memory updater, instead
            # of just passing a list.
            # We are going from two inputs of (batch_size, num_options, encoding_dim) to one input of
            # (batch_size, num_options, encoding_dim * 2).
            updater_input = merge([current_memory, attended_knowledge],
                                  mode='concat',
                                  concat_axis=2)
            memory_updater = TimeDistributed(self.memory_updater(encoding_dim=self.embedding_size),
                                             name='memory_updater_%d' % i)
            current_memory = memory_updater(updater_input)

        # Step 5: Finally, run the sentence encoding, the current memory, and the attended
        # background knowledge through an entailment model to get a final true/false score.
        entailment_input = merge([encoded_proposition, current_memory, attended_knowledge],
                                 mode='concat',
                                 concat_axis=2)
        combined_input = TimeDistributed(self.entailment_combiner,
                                         name="entailment_combiner")(entailment_input)
        entailment_output = self.entailment_model.classify(combined_input, multiple_choice=True)

        # Step 6: Define the model, and return it. The model will be compiled and trained by the
        # calling method.
        memory_network = Model(input=[proposition_input_layer, knowledge_input_layer],
                               output=entailment_output)
        return memory_network

    @overrides
    def evaluate(self, labels, test_input):
        """
        We need to override this method, because our test input is already grouped by question.
        """
        scores = self.model.evaluate(test_input, labels)
        return scores[1]  # NOTE: depends on metrics=['accuracy'] in self.model.compile()

    @overrides
    def _get_training_data(self):
        dataset = TextDataset.read_from_file(self.train_file, tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset,
                                                                   self.train_background,
                                                                   tokenizer=self.tokenizer)
        self.data_indexer.fit_word_dictionary(background_dataset)
        question_dataset = background_dataset.to_question_dataset()
        if self.max_training_instances is not None:
            question_dataset = question_dataset.truncate(self.max_training_instances)
        self.training_dataset = background_dataset
        return self.prep_labeled_data(question_dataset, for_train=True, shuffle=True)

    @overrides
    def _get_validation_data(self):
        dataset = TextDataset.read_from_file(self.validation_file, tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset,
                                                                   self.validation_background,
                                                                   tokenizer=self.tokenizer)
        question_dataset = background_dataset.to_question_dataset()
        self.validation_dataset = question_dataset
        return self.prep_labeled_data(question_dataset, for_train=False, shuffle=True)

    @overrides
    def _get_test_data(self):
        dataset = TextDataset.read_from_file(self.test_file)
        background_dataset = TextDataset.read_background_from_file(dataset,
                                                                   self.test_background,
                                                                   tokenizer=self.tokenizer)
        question_dataset = background_dataset.to_question_dataset()
        return self.prep_labeled_data(question_dataset, for_train=False, shuffle=True)

    @overrides
    def _get_debug_dataset_and_input(self):
        dataset = TextDataset.read_from_file(self.debug_file)
        background_dataset = TextDataset.read_background_from_file(dataset,
                                                                   self.debug_background,
                                                                   tokenizer=self.tokenizer)
        question_dataset = background_dataset.to_question_dataset()
        inputs, _ = self.prep_labeled_data(question_dataset, for_train=False, shuffle=False)
        return question_dataset, inputs

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
        all_question_scores = self.score(debug_inputs)
        all_question_attention_outputs = self.debug_model.predict(debug_inputs)
        if self.num_memory_layers == 1:
            all_question_attention_outputs = [all_question_attention_outputs]
        # Collect values from all hops of attention for a given instance into attention_values.
        for instance, question_scores, *question_attention_values in zip(debug_dataset.instances,
                                                                         all_question_scores,
                                                                         *all_question_attention_outputs):
            label = instance.label
            print("Correct answer: %s" % label, file=debug_output_file)
            for option_id, option_instance in enumerate(instance.options):
                option_sentence = option_instance.text
                option_background_info = option_instance.background
                option_score = question_scores[option_id]
                # Remove the attention values for padding
                option_attention_values = [hop_attention_values[option_id]
                                           for hop_attention_values in question_attention_values]
                option_attention_values = [values[-len(option_background_info):]
                                           for values in option_attention_values]
                print("\tOption %d: %s" % (option_id, option_sentence), file=debug_output_file)
                print("\tAssigned score: %.4f" % option_score, file=debug_output_file)
                print("\tWeights on background:", file=debug_output_file)
                for i, background_i in enumerate(option_background_info):
                    if i >= len(option_attention_values[0]):
                        # This happens when IndexedBackgroundInstance.pad() ignored some
                        # sentences (at the end). Let's ignore them too.
                        break
                    all_hops_attention_i = ["%.4f" % values[i] for values in option_attention_values]
                    print("\t\t%s\t%s" % (" ".join(all_hops_attention_i), background_i),
                          file=debug_output_file)
                print(file=debug_output_file)
        debug_output_file.close()
