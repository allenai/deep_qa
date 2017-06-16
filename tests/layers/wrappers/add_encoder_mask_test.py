# pylint: disable=no-self-use,invalid-name
import numpy
from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers.wrappers import AddEncoderMask, OutputMask
from deep_qa.testing.test_case import DeepQaTestCase
from deep_qa.training.models import DeepQaModel
from keras.layers import Embedding, Input


class TestAddEncoderMask(DeepQaTestCase):
    def test_mask_is_computed_correctly(self):
        background_input = Input(shape=(None, 3), dtype='int32')
        embedding = Embedding(input_dim=3, output_dim=2, mask_zero=True)
        embedded_background = embedding(background_input)
        encoded_background = BOWEncoder(units=2)(embedded_background)
        encoded_background_with_mask = AddEncoderMask()([encoded_background, embedded_background])

        mask_output = OutputMask()(encoded_background_with_mask)
        model = DeepQaModel(inputs=[background_input], outputs=mask_output)

        test_background = numpy.asarray([
                [
                        [0, 0, 0],
                        [2, 2, 2],
                        [0, 0, 0],
                        [0, 1, 2],
                        [1, 0, 0],
                        [0, 0, 0],
                        [0, 1, 0],
                        [1, 1, 1],
                ]
        ])
        expected_mask = numpy.asarray([[0, 1, 0, 1, 1, 0, 1, 1]])
        actual_mask = model.predict([test_background])
        numpy.testing.assert_array_equal(expected_mask, actual_mask)
