from copy import deepcopy
from typing import Any, Dict, List
from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Dropout, merge
from keras.models import Model

from ..common.params import get_choice_with_default
from ..data.dataset import Dataset, IndexedDataset, TextDataset  # pylint: disable=unused-import
from ..data.text_instance import TrueFalseInstance
from ..layers.knowledge_selectors import selectors
from ..layers.memory_updaters import updaters
from ..layers.entailment_models import entailment_models, entailment_input_combiners
from .nn_solver import NNSolver


# TODO(matt): make this class abstract, and make a TrueFalseMemoryNetwork subclass.
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

    # This specifies whether the entailment decision made my this solver (if any) has a sigmoid
    # activation or a softmax activation.  This value is read by some pre-trainers, which need
    # to know how to construct data for training a model.  Because it's necessary for pre-training,
    # and we need to be able to override it in subclasses, it doesn't really work to set this in
    # the constructor, so we make it a class variable instead.
    has_sigmoid_entailment = False

    def __init__(self, params: Dict[str, Any]):

        self.train_background = params.pop('train_background', None)
        self.validation_background = params.pop('validation_background', None)

        self.num_memory_layers = params.pop('num_memory_layers', 1)

        # These parameters specify the kind of knowledge selector, used to compute an attention
        # over a collection of background information.
        # If given, this must be a dict.  We will use the "type" key in this dict (which must match
        # one of the keys in `selectors`) to determine the type of the selector, then pass the
        # remaining args to the selector constructor.
        self.knowledge_selector_params = params.pop('knowledge_selector', {})

        # Same deal with these three as with the knowledge selector params.
        self.memory_updater_params = params.pop('memory_updater', {})
        self.entailment_combiner_params = params.pop('entailment_input_combiner', {})
        self.entailment_model_params = params.pop('entailment_model', {})

        # Now that we've processed all of our parameters, we can call the superclass constructor.
        # The superclass will check that there are no unused parameters, so we need to call this
        # _after_ we've popped everything we use.
        super(MemoryNetworkSolver, self).__init__(params)
        self.name = "MemoryNetworkSolver"

        # These are the entailment models that are compatible with this solver.
        self.entailment_choices = ['true_false_mlp']

        # Model-specific variables that will get set and used later.  For many of these, we don't
        # want to set them now, because they use max length information that only gets set after
        # reading the training data.
        self.knowledge_selector_layers = {}
        self.memory_updater_layers = {}
        self.entailment_input_combiner = None
        self.entailment_model = None
        self.max_knowledge_length = None

    @overrides
    def can_train(self) -> bool:
        return self.train_background is not None and super(MemoryNetworkSolver, self).can_train()

    @overrides
    def _get_training_files(self):
        return [self.train_file, self.train_background]

    @overrides
    def _get_validation_files(self):
        if self.validation_file:
            return [self.validation_file, self.validation_background]
        else:
            return None

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = super(MemoryNetworkSolver, self)._load_dataset_from_files(files)
        return TextDataset.read_background_from_file(dataset, files[1])

    @overrides
    def _instance_type(self):
        return TrueFalseInstance

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(MemoryNetworkSolver, cls)._get_custom_objects()
        for object_dict in [updaters, selectors, entailment_input_combiners]:
            for value in object_dict.values():
                custom_objects[value.__name__] = value
        return custom_objects

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'background_sentences': self.max_knowledge_length,
                }

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_knowledge_length = max_lengths['background_sentences']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][1]
        self.max_knowledge_length = self.model.get_input_shape_at(0)[1][1]

    def _get_question_shape(self):
        """
        This is the shape of the input word sequences for a question, not including the batch size.
        """
        return (self.max_sentence_length,)

    def _get_background_shape(self):
        """
        This is the shape of background data (word sequences) associated with a question, not
        including the batch size.
        """
        return (self.max_knowledge_length, self.max_sentence_length)

    def _get_knowledge_axis(self):
        """
        We need to merge and concatenate things in the course of the memory network, and we do it
        in the knowledge_length dimension.  This tells us which axis that dimension is in,
        including the batch_size.

        So, for the true/false memory network, which has background input shape
        (batch_size, knowledge_length, sentence_length), this would be 1.  For the multiple choice
        memory network, which has background input shape
        (batch_size, num_options, knowledge_length, sentence_length), this would be 2.
        """
        # pylint: disable=no-self-use
        return 1

    def _get_merged_background_shape(self):
        """
        This method returns a lambda function, which takes input the shape of the question encoding
        and the knowledge encoding, and returns as output the shape of the merged question and
        background encodings.  This merge just stacks the question encoding on top of the
        background encoding, adding one to the knowledge_length dimension.
        """
        knowledge_axis = self._get_knowledge_axis()
        def merged_shape(input_shapes):
            background_shape = [x for x in input_shapes[1]]
            background_shape[knowledge_axis] += 1
            return tuple(background_shape)
        return merged_shape

    def _get_weighted_average_shape(self):
        """
        Similar to _get_merged_background_shape, this method returns the shape of a function that
        averages over the knowledge axis.  All we have to do is drop the knowledge axis from the
        shape.
        """
        knowledge_axis = self._get_knowledge_axis()
        def merged_shape(input_shapes):
            shape = [x for x in input_shapes[0]]
            shape.pop(knowledge_axis)
            return tuple(shape)
        return merged_shape

    def _get_knowledge_selector(self, layer_num: int):
        """
        Instantiates a KnowledgeSelector layer.  This is an overridable method because some
        subclasses might need to TimeDistribute this, for instance.
        """
        if layer_num not in self.knowledge_selector_layers:
            layer = self._get_new_knowledge_selector(name='knowledge_selector_%d' % layer_num)
            self.knowledge_selector_layers[layer_num] = layer
        return self.knowledge_selector_layers[layer_num]

    def _get_new_knowledge_selector(self, name='knowledge_selector'):
        # The code that follows would be destructive to self.knowledge_selector_params (lots of
        # calls to params.pop()), but it's possible we'll want to call this more than once.  So
        # we'll make a copy and use that instead of self.knowledge_selector_params.
        params = deepcopy(self.knowledge_selector_params)
        selector_type = get_choice_with_default(params, "type", list(selectors.keys()))
        params['name'] = name
        return selectors[selector_type](**params)

    def _get_memory_updater(self, layer_num: int):
        """
        Instantiates a MemoryUpdater layer.  This is an overridable method because some subclasses
        might need to TimeDistribute this, for instance.
        """
        if layer_num not in self.memory_updater_layers:
            layer = self._get_new_memory_updater(name='memory_updater_%d' % layer_num)
            self.memory_updater_layers[layer_num] = layer
        return self.memory_updater_layers[layer_num]

    def _get_new_memory_updater(self, name='memory_updater'):
        # The code that follows would be destructive to self.memory_updater_params (lots of calls
        # to params.pop()), but it's possible we'll want to call this more than once.  So we'll
        # make a copy and use that instead of self.memory_updater_params.
        params = deepcopy(self.memory_updater_params)
        updater_type = get_choice_with_default(params, "type", list(updaters.keys()))
        params['name'] = name
        params['encoding_dim'] = self.embedding_size
        return updaters[updater_type](**params)

    def _get_entailment_input_combiner(self):
        """
        Instantiates an EntailmentCombiner layer.  This is an overridable method because some
        subclasses might need to TimeDistribute this, for instance.
        """
        if self.entailment_input_combiner is None:
            self.entailment_input_combiner = self._get_new_entailment_input_combiner()
        return self.entailment_input_combiner

    def _get_new_entailment_input_combiner(self):
        # The code that follows would be destructive to self.entailment_combiner_params (lots of
        # calls to params.pop()), but it's possible we'll want to call this more than once.  So
        # we'll make a copy and use that instead of self.entailment_combiner_params.
        params = deepcopy(self.entailment_combiner_params)
        params['encoding_dim'] = self.embedding_size
        combiner_type = get_choice_with_default(params, "type", list(entailment_input_combiners.keys()))
        return entailment_input_combiners[combiner_type](**params)

    def _get_entailment_output(self, combined_input):
        """
        Gets from the combined entailment input to an output that matches the training labels.
        This is typically done using self.entailment_model.classify(), but could do other things
        also.

        To allow for subclasses to take additional inputs in the entailment model, the return value
        is a tuple of ([additional input layers], output layer).  For instance, this is where
        answer options go, for models that separate the question text from the answer options.
        """
        return [], self._get_entailment_model().classify(combined_input)

    def _get_entailment_model(self):
        if self.entailment_model is None:
            self.entailment_model = self._get_new_entailment_model()
        return self.entailment_model

    def _get_new_entailment_model(self):
        # The code that follows would be destructive to self.entailment_model_params (lots of calls
        # to params.pop()), but it's possible we'll want to call this more than once.  So we'll
        # make a copy and use that instead of self.entailment_model_params.
        entailment_params = deepcopy(self.entailment_model_params)
        model_type = get_choice_with_default(entailment_params, "type", self.entailment_choices)
        # TODO(matt): Not great to have these two lines here.
        if model_type == 'question_answer_mlp':
            entailment_params['answer_dim'] = self.embedding_size
        return entailment_models[model_type](entailment_params)

    @overrides
    def _build_model(self):
        # Steps 1 and 2: Convert inputs to sequences of word vectors, for both the question
        # inputs and the knowledge inputs.
        question_input_layer, question_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_question_shape(), name_prefix="sentence")
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_background_shape(), name_prefix="background")

        # Step 3: Encode the two embedded inputs using the sentence encoder.
        question_encoder = self._get_sentence_encoder()

        # Knowledge encoder will have the same encoder running on a higher order tensor.
        # i.e., question_encoder: (samples, num_words, word_dim) -> (samples, word_dim)
        # and knowledge_encoder: (samples, knowledge_len, num_words, word_dim) ->
        #                       (samples, knowledge_len, word_dim)
        # TimeDistributed generally loops over the second dimension.
        knowledge_encoder = TimeDistributed(question_encoder, name='knowledge_encoder')
        encoded_question = question_encoder(question_embedding)  # (samples, word_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)  # (samples, knowledge_len, word_dim)

        # Step 4: Merge the two encoded representations and pass into the knowledge backed scorer.
        # At each step in the following loop, we take the question encoding, or the output of
        # the previous memory layer, merge it with the knowledge encoding and pass it to the
        # current memory layer.
        current_memory = encoded_question

        knowledge_axis = self._get_knowledge_axis()
        for i in range(self.num_memory_layers):
            # We want to merge a matrix and a tensor such that the new tensor will have one
            # additional row (at the beginning) in all slices.
            # (samples, word_dim) + (samples, knowledge_len, word_dim)
            #       -> (samples, 1 + knowledge_len, word_dim)
            # Since this is an unconventional merge, we define a customized lambda merge.
            # Keras cannot infer the shape of the output of a lambda function, so we make
            # that explicit.
            merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=knowledge_axis),
                                                           layer_outs[1]],
                                                          axis=knowledge_axis)
            merged_shape = self._get_merged_background_shape()
            merged_encoded_rep = merge([current_memory, encoded_knowledge],
                                       mode=merge_mode,
                                       output_shape=merged_shape,
                                       name='concat_question_with_background_%d' % i)

            # Regularize it
            regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
            knowledge_selector = self._get_knowledge_selector(i)
            attention_weights = knowledge_selector(regularized_merged_rep)
            # Defining weighted average as a custom merge mode. Takes two inputs: data and weights
            # ndim of weights is one less than data.
            weighted_average = lambda avg_inputs: K.sum(avg_inputs[0] * K.expand_dims(avg_inputs[1], dim=-1),
                                                        axis=knowledge_axis)
            # input shapes: (samples, knowledge_len, input_dim), (samples, knowledge_len)
            # output shape: (samples, input_dim)
            weighted_average_shape = self._get_weighted_average_shape()
            attended_knowledge = merge([encoded_knowledge, attention_weights],
                                       mode=weighted_average,
                                       output_shape=weighted_average_shape,
                                       name='background_weighted_average_%d' % i)

            # To make this easier to TimeDistribute, we're going to concatenate the current memory
            # with the attended knowledge, and use that as the input to the memory updater, instead
            # of just passing a list.
            # We going from two inputs of (batch_size, encoding_dim) to one input of (batch_size,
            # encoding_dim * 2).
            updater_input = merge([current_memory, attended_knowledge],
                                  mode='concat',
                                  concat_axis=knowledge_axis,
                                  name='concat_current_memory_with_background_%d' % i)
            memory_updater = self._get_memory_updater(i)
            current_memory = memory_updater(updater_input)


        # TODO(matt): we now have subclasses that do answer selection, instead of entailment, and
        # it's not very nice to shoehorn them into the same "entailment" model.  It would be nice
        # to generalize this into some "final output" section, but I'm not sure how to do that
        # cleanly.

        # Step 5: Finally, run the sentence encoding, the current memory, and the attended
        # background knowledge through an entailment model to get a final true/false score.
        entailment_input = merge([encoded_question, current_memory, attended_knowledge],
                                 mode='concat',
                                 concat_axis=knowledge_axis,
                                 name='concat_entailment_inputs')
        combined_input = self._get_entailment_input_combiner()(entailment_input)
        extra_entailment_inputs, entailment_output = self._get_entailment_output(combined_input)

        # Step 6: Define the model, and return it. The model will be compiled and trained by the
        # calling method.
        input_layers = [question_input_layer, knowledge_input_layer]
        input_layers.extend(extra_entailment_inputs)
        memory_network = Model(input=input_layers, output=entailment_output)
        return memory_network

    @overrides
    def _handle_debug_output(self, dataset: Dataset, layer_names: List[str], outputs, epoch: int):
        debug_output_file = open("%s_debug_%d.txt" % (self.model_prefix, epoch), "w")
        final_scores = None
        attention_outputs = []
        for i, layer_name in enumerate(layer_names):
            if layer_name.endswith('softmax'):
                final_scores = outputs[i]
            elif 'knowledge_selector' in layer_name:
                attention_outputs.append(outputs[i])
        assert final_scores is not None, "You must include the softmax layer in the debug output!"
        assert len(attention_outputs) > 0, "No attention layer specified; what are you debugging?"
        # Collect values from all hops of attention for a given instance into attention_values.
        for instance, score, *attention_values in zip(dataset.instances, final_scores, *attention_outputs):
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
