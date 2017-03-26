from overrides import overrides

from .masked_layer import MaskedLayer

class Additive(MaskedLayer):
    """
    This ``Layer`` `adds` a parameter value to each cell in the input tensor, similar to a bias
    vector in a ``Dense`` layer, but this `only` adds, one value per cell.  The value to add is
    learned.

    Parameters
    ----------
    initializer: str, optional (default='glorot_uniform')
        Keras initializer for the additive weight.
    """
    def __init__(self, initializer='glorot_uniform', **kwargs):
        super(Additive, self).__init__(**kwargs)

        self.initializer = initializer
        self._additive_weight = None

    @overrides
    def build(self, input_shape):
        super(Additive, self).build(input_shape)
        self._additive_weight = self.add_weight(input_shape[1:],
                                                name='%s_additive' % self.name,
                                                initializer=self.initializer)

    @overrides
    def call(self, inputs, mask=None):
        return inputs + self._additive_weight

    @overrides
    def get_config(self):
        base_config = super(Additive, self).get_config()
        config = {
                'initializer': self.initializer,
                }
        config.update(base_config)
        return config
