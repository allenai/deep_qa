from typing import Any, Dict, List
from overrides import overrides

import numpy
from keras.layers import TimeDistributed
from ...data.instances.true_false_instance import TrueFalseInstance
from ...data.instances.multiple_true_false_instance import MultipleTrueFalseInstance
from ...layers.wrappers import EncoderWrapper
from .memory_network import MemoryNetworkSolver


class MultipleTrueFalseMemoryNetworkSolver(MemoryNetworkSolver):
    '''
    This is a MemoryNetworkSolver that is trained on multiple choice questions, instead of
    true/false questions, where the questions are converted into a collection of true/false
    assertions.

    This needs changes to two areas of the code: (1) how the data is preprocessed, and (2) how the
    model is built.

    Instead of getting a list of positive and negative examples, we get a question with labeled
    answer options, only one of which can be true.  We then pass each option through the same
    basic structure as the MemoryNetworkSolver, and use a softmax at the end to get a final answer.

    This essentially just means making the MemoryNetworkSolver model time-distributed everywhere,
    and adding a final softmax.
    '''

    # See comments in MemoryNetworkSolver for more info on this.
    has_sigmoid_entailment = True
    has_multiple_backgrounds = True

    def __init__(self, params: Dict[str, Any]):
        # Upper limit on number of options per question in the training data. Ignored during
        # testing (we use the value set at training time, either from this parameter or from a
        # loaded model).  If this is not set, we'll calculate a max length from the data.
        self.num_options = params.pop('num_options', None)

        super(MultipleTrueFalseMemoryNetworkSolver, self).__init__(params)

        # Now we set some class-specific member variables.
        self.entailment_choices = ['multiple_choice_mlp']

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'background_sentences': self.max_knowledge_length,
                'num_options': self.num_options,
                }

    @overrides
    def _instance_type(self):
        return TrueFalseInstance

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_knowledge_length = max_lengths['background_sentences']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[0][2]
        self.max_knowledge_length = self.model.get_input_shape_at(0)[1][2]
        self.num_options = self.model.get_input_shape_at(0)[0][1]

    @overrides
    def _get_question_shape(self):
        return (self.num_options, self.max_sentence_length,)

    @overrides
    def _get_background_shape(self):
        return (self.num_options, self.max_knowledge_length, self.max_sentence_length)

    @overrides
    def _get_sentence_encoder(self):
        # TODO(matt): add tests for saving and loading these models, to be sure that these names
        # actually work as expected.  There are currently some Keras bugs stopping those tests from
        # working, though.
        base_sentence_encoder = super(MultipleTrueFalseMemoryNetworkSolver, self)._get_sentence_encoder()
        return EncoderWrapper(base_sentence_encoder, name="timedist_%s" % base_sentence_encoder.name)

    @overrides
    def _get_knowledge_axis(self):
        # pylint: disable=no-self-use
        return 2

    @overrides
    def _get_knowledge_selector(self, layer_num: int):
        base_selector = super(MultipleTrueFalseMemoryNetworkSolver, self)._get_knowledge_selector(layer_num)
        return EncoderWrapper(base_selector, name="timedist_%s" % base_selector.name)

    @overrides
    def _get_knowledge_combiner(self, layer_num: int):
        base_combiner = super(MultipleTrueFalseMemoryNetworkSolver, self)._get_knowledge_combiner(layer_num)
        return TimeDistributed(base_combiner, name="timedist_%s" % base_combiner.name)

    @overrides
    def _get_memory_updater(self, layer_num: int):
        base_memory_updater = super(MultipleTrueFalseMemoryNetworkSolver, self)._get_memory_updater(layer_num)
        return TimeDistributed(base_memory_updater, name="timedist_%s" % base_memory_updater.name)

    @overrides
    def _get_entailment_input_combiner(self):
        base_combiner = super(MultipleTrueFalseMemoryNetworkSolver, self)._get_entailment_input_combiner()
        return TimeDistributed(base_combiner, name="timedist_%s" % base_combiner.name)

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        dataset = super(MultipleTrueFalseMemoryNetworkSolver, self)._load_dataset_from_files(files)
        return dataset.to_question_dataset()

    @overrides
    def _render_layer_outputs(self, instance: MultipleTrueFalseInstance, outputs: Dict[str, numpy.array]) -> str:
        result = ""
        for option_index, option_instance in enumerate(instance.options):
            option_sentence = option_instance.instance.text
            result += "\tOption %d: %s\n" % (option_index, option_sentence)
            if 'entailment_scorer' in outputs:
                result += "\tEntailment score: %.4f\n" % outputs['entailment_scorer'][option_index]
            if any('knowledge_selector' in layer_name for layer_name in outputs.keys()):
                result += "\nWeights on background:\n"
                # In order to use the _render_attention() method in MemoryNetworkSolver, we need
                # to pass only the output that's relevant to this option.  So we create a new dict.
                option_outputs = {}
                for layer_name, output in outputs.items():
                    if self._is_attention_layer(layer_name):
                        option_outputs[layer_name] = output[option_index]
                result += self._render_attention(option_instance, option_outputs, '\t\t')
        return result

    @staticmethod
    def _is_attention_layer(layer_name: str) -> bool:
        if 'knowledge_selector' in layer_name:
            return True
        if layer_name == 'background_input':
            return True
        if layer_name == 'knowledge_encoder':
            return True
        return False
