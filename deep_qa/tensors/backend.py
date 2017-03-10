"""
These are utility functions that are similar to calls to Keras' backend.  Some of these are here
because a current function in keras.backend is broken, some are things that just haven't been
implemented.
"""
import keras.backend as K

VERY_LARGE_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_NEGATIVE_NUMBER = -VERY_LARGE_NUMBER

def switch(cond, then_tensor, else_tensor):
    """
    Keras' implementation of K.switch currently uses tensorflow's switch function, which only
    accepts scalar value conditions, rather than boolean tensors which are treated in an
    elementwise function.  This doesn't match with Theano's implementation of switch, but using
    tensorflow's select, we can exactly retrieve this functionality.
    """

    if K.backend() == 'tensorflow':
        import tensorflow as tf
        cond_shape = cond.get_shape()
        input_shape = then_tensor.get_shape()
        if cond_shape[-1] != input_shape[-1] and cond_shape[-1] == 1:
            # This happens when the last dim in the input is an embedding dimension. Keras usually does not
            # mask the values along that dimension. Theano broadcasts the value passed along this dimension,
            # but TF does not. Using K.dot() since cond can be a tensor.
            cond = K.dot(tf.cast(cond, tf.float32), tf.ones((1, input_shape[-1])))
        return tf.select(tf.cast(cond, dtype=tf.bool), then_tensor, else_tensor)
    else:
        import theano.tensor as T
        return T.switch(cond, then_tensor, else_tensor)


def cumulative_sum(tensor, axis=-1):
    """
    Keras' backend does not have tf.cumsum().  We're adding it here.
    """
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.cumsum(tensor, axis=axis)
    else:
        import theano.tensor as T
        return T.cumsum(tensor, axis=axis)


def very_negative_like(tensor):
    return K.ones_like(tensor) * VERY_NEGATIVE_NUMBER


def last_dim_flatten(input_tensor):
    '''
    Takes a tensor and returns a matrix while preserving only the last dimension from the input.
    '''
    input_ndim = K.ndim(input_tensor)
    shuffle_pattern = (input_ndim - 1,) + tuple(range(input_ndim - 1))
    dim_shuffled_input = K.permute_dimensions(input_tensor, shuffle_pattern)
    return K.transpose(K.batch_flatten(dim_shuffled_input))


def tile_vector(vector, matrix):
    """
    NOTE: If your matrix has known shape (i.e., the relevant dimension from `K.int_shape(matrix) is
    not None`), you should just use `K.repeat_elements(vector)` instead of this.  This method
    works, however, when the number of rows in your matrix is unknown at graph compilation time.

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
    NOTE: If your vector has known shape (i.e., the relevant dimension from `K.int_shape(vector) is
    not None`), you should just use `K.repeat_elements(scalar)` instead of this.  This method
    works, however, when the number of entries in your vector is unknown at graph compilation time.

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


def apply_feed_forward(input_tensor, weights, activation):
    '''
    Takes an input tensor, sequence of weights and an activation and builds an MLP.
    This can also be achieved by defining a sequence of Dense layers in Keras, but doing this
    might be desirable if the operation needs to be done within the call method of a more complex
    layer. Moreover, we are not applying biases here. The input tensor can have any number of
    dimensions. But the last dimension, and the sequence of weights are expected to be compatible.
    '''
    current_tensor = input_tensor
    for weight in weights:
        current_tensor = activation(K.dot(current_tensor, weight))
    return current_tensor


def l1_normalize(tensor_to_normalize, mask=None):
    """
    Normalize a tensor by its L1 norm. Takes an optional mask.

    When the vector to be normalized is all 0's we return the uniform
    distribution (taking masking into account, so masked values are still 0.0).
    When the vector to be normalized is completely masked, we return the
    uniform distribution over the max padding length of the tensor.

    See the tests for concrete examples of the aforementioned behaviors.

    Parameters
    ----------
    tensor_to_normalize : Tensor
        Tensor of shape (batch size, x) to be normalized, where
        x is arbitrary.

    mask: Tensor, optional
        Tensor of shape (batch size, x) indicating which elements
        of ``tensor_to_normalize`` are padding and should
        not be considered when normalizing.

    Returns
    -------
    normalized_tensor : Tensor
        Normalized tensor with shape (batch size, x).
    """
    if mask is None:
        mask = K.ones_like(tensor_to_normalize)

    # We cast the  mask to float32 to prevent dtype
    # issues when multiplying it with other things
    mask = K.cast(mask, "float32")

    # We apply the mask to the tensor and take the sum
    # of the values in each row.
    row_sum = K.sum(mask * tensor_to_normalize, axis=-1, keepdims=True)

    # We divide the tensor by the sum of the elements in the rows,
    # and then apply the mask. This is the result a naive
    # implementation would yield; we instead return the uniform distribution
    # in a host of special cases (see the docstring and tests for more detail).
    normal_result = (tensor_to_normalize / row_sum) * mask

    mask_row_sum = K.sum(mask, axis=1, keepdims=True)

    # The number of non-masked elements in the tensor to normalize.
    # If all the elements in the tensor to normalize are masked,
    # we set it to be the number of elements in the tensor to normalize.
    divisor = K.sum(switch(mask_row_sum, mask, K.ones_like(mask)), axis=1,
                    keepdims=True)

    # This handles the case where mask is all 0 and all values are 0.
    # If the sum of mask_row_sum and row_sum is 0, make mask all ones,
    # else just keep the mask as it is.
    temp_mask = switch(mask_row_sum+row_sum, mask, K.ones_like(mask))

    uniform = (K.ones_like(mask)/(divisor)) * temp_mask
    normalized_tensors = switch(row_sum, normal_result, uniform)
    return normalized_tensors
