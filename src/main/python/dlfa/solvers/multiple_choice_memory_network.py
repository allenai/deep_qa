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
        self.instance_type = 'multiple_choice'

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
    def _get_sentence_shape(self):
        return (self.num_options, self.max_sentence_length,)

    @overrides
    def _get_background_shape(self):
        return (self.num_options, self.max_knowledge_length, self.max_sentence_length)

    @overrides
    def _get_sentence_encoder(self):
        return TimeDistributed(super(MultipleChoiceMemoryNetworkSolver, self)._get_sentence_encoder())

    @overrides
    def _get_merge_axis(self):
        return 2

    @overrides
    def _get_merged_background_shape(self):
        return lambda layer_out_shapes: (layer_out_shapes[1][0],
                                         layer_out_shapes[1][1],
                                         layer_out_shapes[1][2] + 1,
                                         layer_out_shapes[1][3])

    @overrides
    def _get_knowledge_selector(self, layer_num: int):
        return TimeDistributed(super(MultipleChoiceMemoryNetworkSolver, self)._get_knowledge_selector())

    @overrides
    def _get_weighted_average_shape(self):
        return lambda input_shapes: (input_shapes[0][0], input_shapes[0][1], input_shapes[0][3])

    @overrides
    def _get_memory_updater(self, layer_num: int):
        return TimeDistributed(super(MultipleChoiceMemoryNetworkSolver, self)._get_memory_updater())

    @overrides
    def _get_entailment_combiner(self):
        return TimeDistributed(super(MultipleChoiceMemoryNetworkSolver, self)._get_entailment_combiner())

    @overrides
    def _get_entailment_output(self, combined_input):
        return self.entailment_model.classify(combined_input, multiple_choice=True)

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

    @overrides
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
