from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class Repeat(MaskedLayer):
    """
    This `Layer` calls `K.repeat_elements` on both the input and the mask, after calling
    `K.expand_dims`.

    If the mask is not `None`, we must be able to call `K.expand_dims` using the same axis
    parameter as we do for the input.

    Input:
        - A tensor of arbitrary shape.

    Output:
        - The input tensor repeated along one of the dimensions.
    """
    def __init__(self, axis: int, repetitions: int, **kwargs):
        self.axis = axis
        self.repetitions = repetitions
        super(Repeat, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        return K.repeat_elements(K.expand_dims(mask, self.axis), self.repetitions, self.axis)

    @overrides
    def compute_output_shape(self, input_shape):
        return input_shape[:self.axis] + (self.repetitions,) + input_shape[self.axis:]

    @overrides
    def call(self, inputs, mask=None):
        return K.repeat_elements(K.expand_dims(inputs, self.axis), self.repetitions, self.axis)

    @overrides
    def get_config(self):
        base_config = super(Repeat, self).get_config()
        config = {'axis': self.axis, 'repetitions': self.repetitions}
        config.update(base_config)
        return config
