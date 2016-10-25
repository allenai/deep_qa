from typing import Any, Dict

from overrides import overrides

from keras.layers import merge, Dense, TimeDistributed

from ...data.instances.sentence_pair_instance import SentencePairInstance
from ...training.pretraining.text_pretrainer import TextPretrainer
from ...training.models import DeepQaModel


class EncoderPretrainer(TextPretrainer):
    """
    Here we pretrain the encoder (including the embedding if needed) using an objective similar to word2vec.
    The input data to this pretrainer is labeled pairs of sentences with the label indicating whether the
    sentences co-occur in the same context or not. The encoded representations of the two sentences are fed to a
    MLP which is trained along with the encoder with the objective of minimizing the categorical crossentropy.
    """
    def __init__(self, trainer, params: Dict[str, Any]):
        super(EncoderPretrainer, self).__init__(trainer, params)
        self.name = "EncoderPretrainer"
        self.loss = "binary_crossentropy"

    @overrides
    def _instance_type(self):
        return SentencePairInstance

    @overrides
    def _build_model(self):
        # pylint: disable=protected-access
        sentence_shape = (self.trainer.max_sentence_length,)
        first_sentence_input, embedded_first_sentence = self.trainer._get_embedded_sentence_input(sentence_shape,
                                                                                                  "first_sentence")
        second_sentence_input, embedded_second_sentence = self.trainer._get_embedded_sentence_input(
                sentence_shape, "second_sentence")
        sentence_encoder = self.trainer._get_sentence_encoder()
        while isinstance(sentence_encoder, TimeDistributed):
            sentence_encoder = sentence_encoder.layer
        encoded_first_sentence = sentence_encoder(embedded_first_sentence)
        encoded_second_sentence = sentence_encoder(embedded_second_sentence)
        merged_sentences = merge([encoded_first_sentence, encoded_second_sentence], mode='concat')
        # The activation is a sigmoid because the label is one integer instead of a one-hot array, and it
        # does not make sense to do a softmax on one value.
        binary_prediction = Dense(1, activation='sigmoid', name='context_prediction')(merged_sentences)
        return DeepQaModel(input=[first_sentence_input, second_sentence_input], output=binary_prediction)
