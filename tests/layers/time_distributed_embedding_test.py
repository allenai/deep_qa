# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input
from keras import backend as K

from deep_qa.layers import TimeDistributedEmbedding
from ..common.test_case import DeepQaTestCase


class TestTimeDistributedEmbeddings(DeepQaTestCase):
    def test_time_distributed_embedding_masking(self):
        input_layer = Input(shape=(2, 3), dtype='int32')
        embedding = TimeDistributedEmbedding(input_dim=3, output_dim=5, mask_zero=True)
        embedding(input_layer)  # A call to the layer is required to define the mask.
        embedding_mask = embedding.get_output_mask_at(0)
        get_mask = K.function([input_layer], [embedding_mask])
        input_val = numpy.asarray([[[1, 0, 2], [0, 2, 1]]])
        mask_val = get_mask([input_val])[0]
        assert numpy.all(mask_val == numpy.asarray([[1, 0, 1], [0, 1, 1]], dtype='int8'))
