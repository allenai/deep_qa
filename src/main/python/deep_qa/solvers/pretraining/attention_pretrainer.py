from overrides import overrides

from keras import backend as K
from keras.layers import merge, Dropout, TimeDistributed
from keras.models import Model

from ...training.pretraining.pretrainer import Pretrainer
from ...data.dataset import TextDataset


# TODO(matt): This needs to be merged with the Trainer stuff.
class AttentionPretrainer(Pretrainer):
    # While it's not great, we need access to a few of the internals of the solver, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, solver, instance_file, background_file, **kwargs):
        super(AttentionPretrainer, self).__init__(solver, **kwargs)
        self.instance_file = instance_file
        self.background_file = background_file

    @overrides
    def _load_dataset(self):
        dataset = TextDataset.read_from_file(self.instance_file,
                                             self.solver._instance_type(),
                                             tokenizer=self.solver.tokenizer)
        return TextDataset.read_labeled_background_from_file(dataset, self.background_file)

    @overrides
    def _get_model(self):
        """
        This model basically just pulls out the first half of the memory network model, up until
        the first attention layer.

        Because the solver we're pretraining might have some funny input shapes, we don't use
        solver._get_question_shape() directly; instead we re-create it for the case where we don't
        have TimeDistributed input.
        """
        # What follows is a lightly-edited version of the code from
        # MemoryNetworkSolver._build_model().
        sentence_shape = (self.solver.max_sentence_length,)
        background_shape = (self.solver.max_knowledge_length, self.solver.max_sentence_length)

        sentence_input_layer, sentence_embedding = self.solver._get_embedded_sentence_input(
                input_shape=sentence_shape, name_prefix="sentence")
        background_input_layer, background_embedding = self.solver._get_embedded_sentence_input(
                input_shape=background_shape, name_prefix="background")

        sentence_encoder = self.solver._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer

        background_encoder = TimeDistributed(sentence_encoder, name='background_encoder')
        encoded_sentence = sentence_encoder(sentence_embedding)  # (samples, word_dim)
        encoded_background = background_encoder(background_embedding)  # (samples, background_len, word_dim)

        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=1),
                                                       layer_outs[1]],
                                                      axis=1)
        merged_encoded_rep = merge([encoded_sentence, encoded_background],
                                   mode=merge_mode,
                                   output_shape=(self.solver.max_knowledge_length + 1,
                                                 self.solver.max_sentence_length),
                                   name='concat_sentence_with_background_%d' % 0)

        regularized_merged_rep = Dropout(0.2)(merged_encoded_rep)
        knowledge_selector = self.solver._get_knowledge_selector(0)
        while isinstance(knowledge_selector, TimeDistributed):
            knowledge_selector = knowledge_selector.layer
        attention_weights = knowledge_selector(regularized_merged_rep)

        input_layers = [sentence_input_layer, background_input_layer]
        return Model(input=input_layers, output=attention_weights)
