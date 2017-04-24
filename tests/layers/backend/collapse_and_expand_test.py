# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_allclose
from keras.layers import Input, Dense
from keras.models import Model

from deep_qa.layers.backend import CollapseToBatch, ExpandFromBatch, AddMask


class TestCollapseAndExpand:
    # We need to test CollapseToBatch and ExpandFromBatch together, because Keras doesn't like it
    # if you change the batch size between inputs and outputs.  It makes sense to test them
    # together, anyway.
    def test_collapse_and_expand_works_with_dynamic_shape(self):
        batch_size = 3
        length1 = 5
        length2 = 7
        length3 = 2
        dense_units = 6
        input_layer = Input(shape=(length1, None, length3), dtype='float32')
        masked_input = AddMask(mask_value=1)(input_layer)
        collapsed_1 = CollapseToBatch(num_to_collapse=1)(masked_input)
        collapsed_2 = CollapseToBatch(num_to_collapse=2)(masked_input)
        dense = Dense(dense_units)(collapsed_2)
        expanded_1 = ExpandFromBatch(num_to_expand=1)([collapsed_1, masked_input])
        expanded_2 = ExpandFromBatch(num_to_expand=2)([collapsed_2, masked_input])
        expanded_dense = ExpandFromBatch(num_to_expand=2)([dense, masked_input])
        model = Model(inputs=input_layer, outputs=[expanded_1, expanded_2, expanded_dense])

        input_tensor = numpy.random.randint(0, 3, (batch_size, length1, length2, length3))
        expanded_1_tensor, expanded_2_tensor, expanded_dense_tensor = model.predict(input_tensor)
        assert expanded_1_tensor.shape == input_tensor.shape
        assert expanded_2_tensor.shape == input_tensor.shape
        assert expanded_dense_tensor.shape == input_tensor.shape[:-1] + (dense_units,)
        assert_allclose(expanded_1_tensor, input_tensor)
        assert_allclose(expanded_2_tensor, input_tensor)
