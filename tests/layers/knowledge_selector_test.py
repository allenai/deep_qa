# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input, merge
from keras import backend as K

from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers import DotProductKnowledgeSelector, TimeDistributedEmbedding
from deep_qa.layers.wrappers import EncoderWrapper, OutputMask
from deep_qa.training.models import DeepQaModel
from ..common.test_case import DeepQaTestCase


class TestDotProductKnowledgeSelector(DeepQaTestCase):
    def test_on_masked_input(self):
        # TODO(matt): I don't really like having to build the whole model up to the attention
        # component here, but I'm not sure how to just test the selector with the right mask
        # without going through this.
        sentence_input = Input(shape=(3,), dtype='int32')
        background_input = Input(shape=(3, 3), dtype='int32')
        embedding = TimeDistributedEmbedding(input_dim=3, output_dim=2, mask_zero=True)
        embedded_sentence = embedding(sentence_input)
        embedded_background = embedding(background_input)
        encoder = BOWEncoder(units=2)
        encoded_sentence = encoder(embedded_sentence)
        encoded_background = EncoderWrapper(encoder)(embedded_background)
        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], axis=1),
                                                       K.expand_dims(layer_outs[0], axis=1),
                                                       layer_outs[1]],
                                                      axis=1)
        merge_masks = lambda mask_outs: K.concatenate([K.expand_dims(K.zeros_like(mask_outs[1][:, 0]),
                                                                     axis=1),
                                                       K.expand_dims(K.zeros_like(mask_outs[1][:, 0]),
                                                                     axis=1),
                                                       mask_outs[1]], axis=1)

        merged = merge([encoded_sentence, encoded_background],
                       mode=merge_mode,
                       output_shape=(5, 2),
                       output_mask=merge_masks)
        merged_mask = OutputMask()(merged)
        selector = DotProductKnowledgeSelector()
        attention_weights = selector(merged)
        model = DeepQaModel(inputs=[sentence_input, background_input], outputs=[merged_mask, attention_weights])
        model.summary(show_masks=True)

        test_input = numpy.asarray([[2, 2, 2]])
        test_background = numpy.asarray([
                [
                        [2, 2, 2],
                        [2, 2, 2],
                        [0, 0, 0],
                ]
        ])
        expected_mask = numpy.asarray([[0, 0, 1, 1, 0]])
        expected_attention = numpy.asarray([[0.5, 0.5, 0.0]])
        actual_mask, actual_attention = model.predict([test_input, test_background])
        numpy.testing.assert_array_almost_equal(expected_mask, actual_mask)
        numpy.testing.assert_array_almost_equal(expected_attention, actual_attention)
