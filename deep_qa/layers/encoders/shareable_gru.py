from keras import backend as K
from keras.layers import GRU, InputSpec


class ShareableGRU(GRU):
    def __init__(self, *args, **kwargs):
        super(ShareableGRU, self).__init__(*args, **kwargs)

    def call(self, x, mask=None, **kwargs):
        input_shape = K.int_shape(x)
        res = super(ShareableGRU, self).call(x, mask, **kwargs)
        self.input_spec = InputSpec(shape=(self.input_spec.shape[0],
                                           None,
                                           self.input_spec.shape[2]))
        if K.ndim(x) == K.ndim(res):
            # A recent change in Keras
            # (https://github.com/fchollet/keras/commit/a9b6bef0624c67d6df1618ca63d8e8141b0df4d0)
            # made it so that K.rnn with a tensorflow backend does not retain shape information for
            # the sequence length, even if it's present in the input.  We need to fix that here so
            # that our models have the right shape information.  A simple K.reshape is good enough
            # to fix this.
            result_shape = K.int_shape(res)
            if input_shape[1] is not None and result_shape[1] is None:
                shape = (input_shape[0] if input_shape[0] is not None else -1,
                         input_shape[1], result_shape[2])
                res = K.reshape(res, shape=shape)
        return res
