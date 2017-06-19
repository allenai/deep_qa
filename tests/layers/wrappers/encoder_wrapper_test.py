# pylint: disable=no-self-use,invalid-name
import numpy
from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers.wrappers import EncoderWrapper, OutputMask
from deep_qa.testing.test_case import DeepQaTestCase
from deep_qa.training.models import DeepQaModel
from keras.layers import Embedding, Input


class TestEncoderWrapper(DeepQaTestCase):
    def test_mask_is_computed_correctly(self):
        background_input = Input(shape=(3, 3), dtype='int32')
        embedding = Embedding(input_dim=3, output_dim=2, mask_zero=True)
        embedded_background = embedding(background_input)
        encoded_background = EncoderWrapper(BOWEncoder(units=2))(embedded_background)

        mask_output = OutputMask()(encoded_background)
        model = DeepQaModel(inputs=[background_input], outputs=mask_output)

        test_background = numpy.asarray([
                [
                        [0, 0, 0],
                        [2, 2, 2],
                        [0, 0, 0],
                ]
        ])
        expected_mask = numpy.asarray([[0, 1, 0]])
        actual_mask = model.predict([test_background])
        numpy.testing.assert_array_almost_equal(expected_mask, actual_mask)
