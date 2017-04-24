from typing import Dict, List
from overrides import overrides

import numpy
from keras.layers import Layer, TimeDistributed

from ...common.params import Params  # pylint disable: unused-import
from ..memory_networks.memory_network import MemoryNetwork
from ...data.instances.text_classification import TextClassificationInstance
from ...data.instances.multiple_choice_qa import MultipleTrueFalseInstance, convert_dataset_to_multiple_true_false
from ...layers.wrappers import EncoderWrapper


class MultipleTrueFalseMemoryNetwork(MemoryNetwork):
    '''
    This is a MemoryNetwork that is trained on multiple choice questions, instead of
    true/false questions, where the questions are converted into a collection of true/false
    assertions.

    This needs changes to two areas of the code: (1) how the data is preprocessed, and (2) how the
    model is built.

    Instead of getting a list of positive and negative examples, we get a question with labeled
    answer options, only one of which can be true.  We then pass each option through the same
    basic structure as the MemoryNetwork, and use a softmax at the end to get a final answer.

    This essentially just means making the MemoryNetwork model time-distributed everywhere,
    and adding a final softmax.
    '''

    # See comments in MemoryNetwork for more info on this.
    has_sigmoid_entailment = True
    has_multiple_backgrounds = True

    def __init__(self, params: Params):
        # Upper limit on number of options per question in the training data. Ignored during
        # testing (we use the value set at training time, either from this parameter or from a
        # loaded model).  If this is not set, we'll calculate a max length from the data.
        self.num_options = params.pop('num_options', None)

        super(MultipleTrueFalseMemoryNetwork, self).__init__(params)

        # Now we set some class-specific member variables.
        self.entailment_choices = ['multiple_choice_mlp']

    @overrides
    def _get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super(MultipleTrueFalseMemoryNetwork, self)._get_padding_lengths()
        padding_lengths['num_options'] = self.num_options
        return padding_lengths

    @overrides
    def _instance_type(self):
        return TextClassificationInstance

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        super(MultipleTrueFalseMemoryNetwork, self)._set_padding_lengths(padding_lengths)
        if self.num_options is None:
            self.num_options = padding_lengths['num_options']

    @overrides
    def _set_padding_lengths_from_model(self):
        self._set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[0][2:])
        self.max_knowledge_length = self.model.get_input_shape_at(0)[1][2]
        self.num_options = self.model.get_input_shape_at(0)[0][1]

    @overrides
    def _get_question_shape(self):
        return (self.num_options,) + self._get_sentence_shape()

    @overrides
    def _get_background_shape(self):
        return (self.num_options, self.max_knowledge_length) + self._get_sentence_shape()

    @overrides
    def _time_distribute_question_encoder(self, question_encoder: Layer):
        return EncoderWrapper(question_encoder, name="timedist_%s" % question_encoder.name)

    @overrides
    def _get_knowledge_axis(self):
        # pylint: disable=no-self-use
        return 2

    @overrides
    def _get_knowledge_selector(self, layer_num: int):
        base_selector = super(MultipleTrueFalseMemoryNetwork, self)._get_knowledge_selector(layer_num)
        return EncoderWrapper(base_selector, name="timedist_%s" % base_selector.name)

    @overrides
    def _get_knowledge_combiner(self, layer_num: int):
        base_combiner = super(MultipleTrueFalseMemoryNetwork, self)._get_knowledge_combiner(layer_num)
        return TimeDistributed(base_combiner, name="timedist_%s" % base_combiner.name)

    @overrides
    def _get_memory_updater(self, layer_num: int):
        base_memory_updater = super(MultipleTrueFalseMemoryNetwork, self)._get_memory_updater(layer_num)
        return TimeDistributed(base_memory_updater, name="timedist_%s" % base_memory_updater.name)

    @overrides
    def _get_entailment_input_combiner(self):
        base_combiner = super(MultipleTrueFalseMemoryNetwork, self)._get_entailment_input_combiner()
        return TimeDistributed(base_combiner, name="timedist_%s" % base_combiner.name)

    @overrides
    def load_dataset_from_files(self, files: List[str]):
        dataset = super(MultipleTrueFalseMemoryNetwork, self).load_dataset_from_files(files)
        return convert_dataset_to_multiple_true_false(dataset)

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(MultipleTrueFalseMemoryNetwork, cls)._get_custom_objects()
        custom_objects['EncoderWrapper'] = EncoderWrapper
        from ...layers.attention.masked_softmax import MaskedSoftmax
        custom_objects['MaskedSoftmax'] = MaskedSoftmax
        return custom_objects

    @overrides
    def _render_layer_outputs(self, instance: MultipleTrueFalseInstance, outputs: Dict[str, numpy.array]) -> str:
        result = ""
        for option_index, option_instance in enumerate(instance.options):
            option_sentence = option_instance.instance.text
            result += "\tOption %d: %s\n" % (option_index, option_sentence)
            if 'sentence_input' in outputs or 'sentence_encoder' in outputs:
                option_outputs = {}
                for layer_name in ['sentence_input', 'sentence_encoder']:
                    if layer_name in outputs:
                        option_outputs[layer_name] = outputs[layer_name][option_index]
                self._render_instance(option_instance.instance, option_outputs)
            if 'entailment_scorer' in outputs:
                result += "\tEntailment score: %.4f\n" % outputs['entailment_scorer'][option_index]
            if any('knowledge_selector' in layer_name for layer_name in outputs.keys()):
                result += "\nWeights on background:\n"
                # In order to use the _render_attention() method in MemoryNetwork, we need to pass
                # only the output that's relevant to this option.  So we create a new dict.
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
