from keras import backend as K
from keras.layers import Layer
from overrides import overrides


class AddMask(Layer):
    """
    This ``Layer`` adds a mask to a tensor.  It is intended solely for testing, though if you have
    a use case for this outside of testing, feel free to use it.  The ``call()`` method just
    returns the inputs, and the ``compute_mask`` method calls ``K.not_equal(x, mask_value)``, and
    that's it.  This is different from Keras' ``Masking`` layer, which assumes higher-order input
    and does a ``K.any()`` call in ``compute_mask``.

    Input:
        - tensor: a tensor of arbitrary shape

    Output:
        - the same tensor, now with a mask attached of the same shape

    Parameters
    ----------
    mask_value: float, optional (default=0.0)
        This is the value that we will compare to in ``compute_mask``.
    """
    def __init__(self, mask_value: float=0.0, **kwargs):
        self.supports_masking = True
        self.mask_value = mask_value
        super(AddMask, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        return K.cast(K.not_equal(inputs, self.mask_value), 'uint8')

    @overrides
    def get_output_shape_for(self, input_shape):
        return input_shape

    @overrides
    def call(self, x, mask=None):
        # It turns out that Keras doesn't like it if you just return x, so we need to return a
        # different tensor object.  Just doing a cast apparently doesn't work, either, so we'll
        # add 0.
        return x + 0.0

    @overrides
    def get_config(self):
        config = {'mask_value': self.mask_value}
        base_config = super(AddMask, self).get_config()
        config.update(base_config)
        return config
