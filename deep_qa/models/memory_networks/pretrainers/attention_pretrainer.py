import logging
from typing import Any, Dict, List

from overrides import overrides

from keras import backend as K
from keras.layers import merge, Dropout, TimeDistributed

from ..memory_network import MemoryNetwork
from ....common.checks import ConfigurationError
from ....data.dataset import TextDataset
from ....data.instances import instances
from ....layers.wrappers.encoder_wrapper import EncoderWrapper
from ....training.models import DeepQaModel
from ....training.pretraining.text_pretrainer import TextPretrainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class AttentionPretrainer(TextPretrainer):
    """
    This is a generic Pretrainer for the attention mechanism of a MemoryNetwork.

    The data that we take as input here is a train file, just like you would pass to a
    MemoryNetwork, and a background file that has _labeled_ attention (this is different from
    what you typically pass to a MemoryNetwork).  See LabeledBackgroundInstance and
    TextDataset.read_labeled_background_from_file() for more information on the expected input
    here.

    The label we get from a LabeledBackgroundInstance is the expected attention over the
    background sentences.  We use that signal to pretrain the attention component of the memory
    network.

    Because it seems very difficult to get this to train correctly with hard attention, we always
    do pre-training with soft attention, whatever you set for knowledge selector during actual
    training.  We'll set it back to what it was when we're done with pre-training.
    """
    # While it's not great, we need access to a few of the internals of the trainer, so we'll
    # disable protected access checks.
    def __init__(self, trainer, params: Dict[str, Any]):
        if not isinstance(trainer, MemoryNetwork):
            raise ConfigurationError("The AttentionPretrainer needs a subclass of MemoryNetwork")
        instance_type = params.pop('instance_type', None)
        self.__instance_type = instances[instance_type] if instance_type else None
        super(AttentionPretrainer, self).__init__(trainer, params)
        # NOTE: the default here needs to match the default in the KnowledgeSelector classes.
        self._old_hard_attention_setting = self.trainer.knowledge_selector_params.get('hard_selection', False)
        self.trainer.knowledge_selector_params['hard_selection'] = False
        self.name = 'AttentionPretrainer'

    @overrides
    def on_finished(self):
        super(AttentionPretrainer, self).on_finished()
        self.trainer.knowledge_selector_params['hard_selection'] = self._old_hard_attention_setting
        for layer in self.trainer.knowledge_selector_layers.values():
            layer.hard_selection = self._old_hard_attention_setting

    @overrides
    def _instance_type(self):
        # pylint: disable=protected-access
        if self.__instance_type is None:
            self.__instance_type = self.trainer._instance_type()
        return self.__instance_type

    @overrides
    def _load_dataset_from_files(self, files: List[str]):
        """
        This method requires two input files, one with training examples, and one with labeled
        background corresponding to the training examples.

        Note that we're calling TextDataset.read_labeled_background_from_file() here, not
        TextDataset.read_background_from_file(), because we want our Instances to have labeled
        attention for pretraining, not labeled answers.
        """
        dataset = super(AttentionPretrainer, self)._load_dataset_from_files(files)
        return TextDataset.read_labeled_background_from_file(dataset, files[1])

    @overrides
    def _build_model(self):
        """
        This model basically just pulls out the first half of the memory network model, up until
        the first attention layer.

        Because the trainer we're pretraining might have some funny input shapes, we don't use
        trainer._get_question_shape() directly; instead we re-create it for the case where we don't
        have TimeDistributed input.
        """
        # What follows is a lightly-edited version of the code from
        # MemoryNetwork._build_model().
        # pylint: disable=protected-access
        sentence_shape = self.trainer._get_sentence_shape()
        background_shape = (self.trainer.max_knowledge_length,) + self.trainer._get_sentence_shape()

        sentence_input_layer, sentence_embedding = self.trainer._get_embedded_sentence_input(
                input_shape=sentence_shape, name_prefix="pretraining_sentence")
        background_input_layer, background_embedding = self.trainer._get_embedded_sentence_input(
                input_shape=background_shape, name_prefix="pretraining_background")

        sentence_encoder = self.trainer.get_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer

        background_encoder = EncoderWrapper(sentence_encoder, name='pretraining_background_encoder')
        encoded_sentence = sentence_encoder(sentence_embedding)  # (samples, word_dim)
        encoded_background = background_encoder(background_embedding)  # (samples, background_len, word_dim)

        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=1),
                                                       K.expand_dims(layer_outs[1], dim=1),
                                                       layer_outs[2]],
                                                      axis=1)
        def merge_mask(mask_outs):
            return K.concatenate([K.expand_dims(K.zeros_like(mask_outs[2][:, 0]), dim=1),
                                  K.expand_dims(K.zeros_like(mask_outs[2][:, 0]), dim=1),
                                  mask_outs[2]],
                                 axis=1)

        # We don't generate a memory representation here so we just pass another encoded_sentence.
        merged_encoded_rep = merge([encoded_sentence, encoded_sentence, encoded_background],
                                   mode=merge_mode,
                                   output_shape=(self.trainer.max_knowledge_length + 2,
                                                 self.trainer.embedding_dim['words']),
                                   output_mask=merge_mask,
                                   name='pretraining_concat_sentence_with_background_%d' % 0)

        regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
        knowledge_selector = self.trainer._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(regularized_merged_rep)

        input_layers = [sentence_input_layer, background_input_layer]
        return DeepQaModel(input=input_layers, output=attention_weights)
