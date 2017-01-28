from keras.layers import Highway as KerasHighway

class Highway(KerasHighway):
    """
    Keras' `Highway` layer does not support masking, but it easily could, just by returning the
    mask.  This `Layer` makes this possible.
    """
    def __init__(self, **kwargs):
        super(Highway, self).__init__(**kwargs)
        self.supports_masking = True
