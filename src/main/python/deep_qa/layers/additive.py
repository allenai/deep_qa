from keras import initializations
from keras.layers import Layer

class Additive(Layer):
    """
    This Layer _adds_ a parameter value to each cell in the input tensor, similar to a bias vector
    in a Dense layer, but this _only_ adds, one value per cell.  The value to add is learned.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(Additive, self).__init__(**kwargs)

        initialization = kwargs.pop('initialization', 'glorot_uniform')
        self.init = initializations.get(initialization)
        self.initial_weights = kwargs.pop('weights', None)
        self._additive_weights = None

    def build(self, input_shape):
        super(Additive, self).build(input_shape)
        self._additive_weights = self.init(input_shape[1:], name='%s_additive' % self.name)
        self.trainable_weights = [self._additive_weights]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, inputs, mask=None):
        return inputs + self._additive_weights
