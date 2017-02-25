from keras.layers import Layer

class TupleMatch(Layer):
    '''
    ``TupleMatch`` layers take two tuples, which are passed in as a list of two tensor inputs, each of
    shape (batch size, number tuple slots, number of tuple elements), and compare them to determine how
    well they match.  For example, an answer candidate tuple might be compared against a tuple that encodes
    background knowledge to determine how well the background knowledge `entails` the answer candidate.

    While typically the tuples will be the same shape, it need not necessarily be the case.  For example,
    conceivably the two tuples might in the future have differing numbers of words in each slot if different
    tuple creation methods are applied to different text sources.  Alternatively, if broader context is used
    for tuples encoding background knowledge, one of the tuples could potentially have additional slots.
    '''
    def __init__(self, **kwargs):
        super(TupleMatch, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shapes):
        # pylint: disable=unused-argument
        return (input_shapes[0][0], 1)

    def call(self, x, mask=None):
        '''
        Takes two tuples, each of shape (batch size, number tuple slots, number of tuple elements).
        These tuples are provided as a list of two tensors and we return a score indicating the degree
        to which they match.
        '''

        raise NotImplementedError
