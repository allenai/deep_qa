from keras.layers import Layer


class MaskedLayer(Layer):
    """
    Keras 2.0 allowed for arbitrary differences in arguments to the ``call`` method of ``Layers``.
    As part of this, they removed the default ``mask=None`` argument, which means that if you want
    to implement ``call`` with a mask, you need to disable a pylint warning.  Instead of disabling
    it in every single layer in our codebase, which could lead to uncaught errors, we'll have a
    single place where we disable it, and have other layers inherit from this class.
    """
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(MaskedLayer, self).__init__(**kwargs)

    def call(self, inputs, mask=None):  # pylint: disable=arguments-differ
        raise NotImplementedError
