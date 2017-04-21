from overrides import overrides

from keras.layers import Input

from ...data.instances.entailment.snli_instance import SnliInstance
from ...training.text_trainer import TextTrainer
from ...layers.entailment_models import DecomposableAttentionEntailment
from ...training.models import DeepQaModel
from ...common.params import Params


class DecomposableAttention(TextTrainer):
    '''
    This ``TextTrainer`` implements the Decomposable Attention model described in "A Decomposable
    Attention Model for Natural Language Inference", by Parikh et al., 2016, with some optional
    enhancements before the decomposable attention actually happens.  Specifically, Parikh's
    original model took plain word embeddings as input to the decomposable attention; we allow
    other operations the transform these word embeddings, such as running a biLSTM on them, before
    running the decomposable attention layer.

    Inputs:

    - A "text" sentence, with shape (batch_size, sentence_length)
    - A "hypothesis" sentence, with shape (batch_size, sentence_length)

    Outputs:

    - An entailment decision per input text/hypothesis pair, in {entails, contradicts, neutral}.

    Parameters
    ----------
    num_seq2seq_layers : int, optional (default=0)
        After getting a word embedding, how many stacked seq2seq encoders should we use before
        doing the decomposable attention?  The default of 0 recreates the original decomposable
        attention model.
    share_encoders : bool, optional (default=True)
        Should we use the same seq2seq encoder for the text and hypothesis, or different ones?
    decomposable_attention_params : Dict[str, Any], optional (default={})
        These parameters get passed to the
        :class:`~deep_qa.layers.entailment_models.decomposable_attention.DecomposableAttentionEntailment`
        layer object, and control things like the number of output labels, number of hidden layers
        in the entailment MLPs, etc.  See that class for a complete description of options here.
    '''
    def __init__(self, params: Params):
        self.num_seq2seq_layers = params.pop('num_seq2seq_layers', 0)
        self.share_encoders = params.pop('share_encoders', True)
        self.decomposable_attention_params = params.pop('decomposable_attention_params', {})
        super(DecomposableAttention, self).__init__(params)

    @overrides
    def _instance_type(self):
        return SnliInstance

    @overrides
    def _build_model(self):
        text_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="text_input")
        hypothesis_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="hypothesis_input")
        text_embedding = self._embed_input(text_input)
        hypothesis_embedding = self._embed_input(hypothesis_input)

        for i in range(self.num_seq2seq_layers):
            text_encoder_name = "hidden_{}".format(i) if self.share_encoders else "text_{}".format(i)
            text_encoder = self._get_seq2seq_encoder(name=text_encoder_name,
                                                     fallback_behavior="use default params")
            text_embedding = text_encoder(text_embedding)
            hypothesis_encoder_name = "hidden_{}".format(i) if self.share_encoders else "hypothesis_{}".format(i)
            hypothesis_encoder = self._get_seq2seq_encoder(name=hypothesis_encoder_name,
                                                           fallback_behavior="use default params")
            hypothesis_embedding = hypothesis_encoder(hypothesis_embedding)

        entailment_layer = DecomposableAttentionEntailment(**self.decomposable_attention_params)
        entailment_probabilities = entailment_layer([text_embedding, hypothesis_embedding])
        return DeepQaModel(inputs=[text_input, hypothesis_input], outputs=entailment_probabilities)

    @overrides
    def _set_padding_lengths_from_model(self):
        print("Model input shape:", self.model.get_input_shape_at(0))
        self._set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[0][1:])

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(DecomposableAttention, cls)._get_custom_objects()
        custom_objects["DecomposableAttentionEntailment"] = DecomposableAttentionEntailment
        return custom_objects
