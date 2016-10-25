import keras.backend as K


def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only accepts
    scalar value conditions, rather than boolean tensors which are treated in an elementwise function.
    This doesn't match with Theano's implementation of switch, but using tensorflow's select, we can
    exactly retrieve this functionality."""

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.select(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
    else:
        import theano.tensor as T
        return T.switch(cond, then_tensor, else_tensor)


def tile_vector(vector, matrix):
    """
    This method takes a (collection of) vector(s) (shape: (batch_size, vector_dim)), and tiles that
    vector a number of times, giving a matrix of shape (batch_size, tile_length, vector_dim).  (I
    say "vector" and "matrix" here because I'm ignoring the batch_size).  We need the matrix as
    input so we know what the tile_length is - the matrix is otherwise ignored.

    This is necessary in a number of places in the code.  For instance, if you want to do a dot
    product of a vector with all of the vectors in a matrix, the most efficient way to do that is
    to tile the vector first, then do an element-wise product with the matrix, then sum out the
    last mode.  So, we capture this functionality here.

    This is not done as a Keras Layer, however; if you want to use this function, you'll need to do
    it _inside_ of a Layer somehow, either in a Lambda or in the call() method of a Layer you're
    writing.
    """
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (tile_length, batch_size, vector_dim)
    k_ones = K.permute_dimensions(K.ones_like(matrix), [1, 0, 2])

    # Now we have a (tile_length, batch_size, vector_dim)*(batch_size, vector_dim)
    # elementwise multiplication which is broadcast. We then reshape back.
    tiled_vector = K.permute_dimensions(k_ones * vector, [1, 0, 2])
    return tiled_vector

def tile_scalar(scalar, vector):
    """
    This method takes a (collection of) scalar(s) (shape: (batch_size, 1)), and tiles that
    scala a number of times, giving a vector of shape (batch_size, tile_length).  (I say "scalar"
    and "vector" here because I'm ignoring the batch_size).  We need the vector as input so we know
    what the tile_length is - the vector is otherwise ignored.

    This is not done as a Keras Layer, however; if you want to use this function, you'll need to do
    it _inside_ of a Layer somehow, either in a Lambda or in the call() method of a Layer you're
    writing.

    TODO(matt): we could probably make a more general `tile_tensor` method, which can do this for
    any dimenionsality.  There is another place in the code where we do this with a matrix and a
    tensor; all three of these can probably be one function.
    """
    # Tensorflow can't use unknown sizes at runtime, so we have to make use of the broadcasting
    # ability of TF and Theano instead to create the tiled sentence encoding.

    # Shape: (tile_length, batch_size)
    k_ones = K.permute_dimensions(K.ones_like(vector), [1, 0])

    # Now we have a (tile_length, batch_size) * (batch_size, 1) elementwise multiplication which is
    # broadcast. We then reshape back.
    tiled_scalar = K.permute_dimensions(k_ones * K.squeeze(scalar, axis=1), [1, 0])
    return tiled_scalar


def hardmax(unnormalized_attention, knowledge_length):
    """
    A similar operation to softmax, except all of the weight is placed on the mode of the
    distribution.  So, e.g., this function transforms [.34, .2, -1.4] -> [1, 0, 0].

    TODO(matt): we really should have this take an optional mask...
    """
    # (batch_size, knowledge_length)
    tiled_max_values = K.tile(K.expand_dims(K.max(unnormalized_attention, axis=1)), (1, knowledge_length))
    # We now have a matrix where every column in each row has the max knowledge score value from
    # the corresponding row in the unnormalized attention matrix.  Next, we will compare that
    # all-max matrix with the original input, resulting in ones where the column equals max and
    # zero everywhere else.
    # Shape: (batch_size, knowledge_length)
    bool_max_attention = K.equal(unnormalized_attention, tiled_max_values)
    # Needs to be cast to be compatible with TensorFlow
    max_attention = K.cast(bool_max_attention, 'float32')
    return max_attention


def masked_softmax(vector, mask):
    """
    `K.softmax(vector)` does not work if some elements of `vector` should be masked.  This performs
    a softmax on just the non-masked portions of `vector` (passing None in for the mask is also
    acceptable; you'll just get a regular softmax).

    We assume that both `vector` and `mask` (if given) have shape (batch_size, vector_dim).
    """
    exponentiated = K.exp(vector)
    if mask is not None:
        exponentiated = switch(mask, exponentiated, K.zeros_like(exponentiated))
    exp_sum = K.sum(exponentiated, axis=1, keepdims=True)
    return switch(tile_scalar(exp_sum, exponentiated), exponentiated / exp_sum, K.zeros_like(exponentiated))
