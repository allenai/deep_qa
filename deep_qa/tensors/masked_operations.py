from keras import backend as K

from .backend import switch


def masked_batch_dot(tensor_a, tensor_b, mask_a, mask_b):
    '''
    The simplest case where this function is applicable is the following:

    tensor_a: (batch_size, a_length, embed_dim)
    tensor_b: (batch_size, b_length, embed_dim)
    mask_a: None or (batch_size, a_length)
    mask_b: None or (batch_size, b_length)

    Returns:
    a_dot_b: (batch_size, a_length, b_length), with zeros for masked elements.

    This function will also work for larger tensors, as long as `abs(K.ndim(tensor_a) -
    K.ndim(tensor_b)) < 1` (this is due to the limitations of `K.batch_dot`).  We always assume the
    dimension to perform the dot is the last one, and that the masks have one fewer dimension than
    the tensors.
    '''
    if K.ndim(tensor_a) < K.ndim(tensor_b):
        # To simplify the logic below, we'll make sure that tensor_a is always the bigger one.
        tensor_a, tensor_b = tensor_b, tensor_a
        mask_a, mask_b = mask_b, mask_a

    a_dot_axis = K.ndim(tensor_a) - 1
    b_dot_axis = K.ndim(tensor_b) - 1
    if b_dot_axis < a_dot_axis:
        tensor_b = K.expand_dims(tensor_b, axis=-1)

    # (batch_size, a_length, b_length)
    a_dot_b = K.batch_dot(tensor_a, tensor_b, axes=(a_dot_axis, b_dot_axis))
    if b_dot_axis < a_dot_axis:
        a_dot_b = K.squeeze(a_dot_b, axis=-1)

    if mask_a is None and mask_b is None:
        return a_dot_b
    elif mask_a is None:
        # (batch_size, a_length)
        mask_a = K.sum(K.ones_like(tensor_a), axis=-1)
    elif mask_b is None:
        # (batch_size, b_length)
        sum_axis = -1
        if b_dot_axis < a_dot_axis:
            sum_axis -= 1
        mask_b = K.sum(K.ones_like(tensor_b), axis=sum_axis)
    # Casting masks to float since we TF would complain if we multiplied bools.
    float_mask_a = K.cast(mask_a, 'float32')
    float_mask_b = K.cast(mask_b, 'float32')

    if b_dot_axis < a_dot_axis:
        float_mask_b = K.expand_dims(float_mask_b, axis=-1)
    else:
        float_mask_a = K.expand_dims(float_mask_a, axis=-1)
        float_mask_b = K.expand_dims(float_mask_b, axis=-2)
    # (batch_size, a_length, b_length)
    a2b_mask = float_mask_a * float_mask_b

    result = switch(a2b_mask, a_dot_b, K.zeros_like(a_dot_b))
    return result


def masked_softmax(vector, mask):
    """
    `K.softmax(vector)` does not work if some elements of `vector` should be masked.  This performs
    a softmax on just the non-masked portions of `vector` (passing None in for the mask is also
    acceptable; you'll just get a regular softmax).

    We assume that both `vector` and `mask` (if given) have shape (batch_size, vector_dim).

    In the case that the input vector is completely masked, this function returns an array
    of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of a model
    that uses categorial cross-entropy loss.
    """
    # We calculate masked softmax in a numerically stable fashion, as done
    # in https://github.com/rkadlec/asreader/blob/master/asreader/custombricks/softmax_mask_bricks.py
    if mask is not None:
        # Here we get normalized log probabilities for
        # enhanced numerical stability.
        mask = K.cast(mask, "float32")
        input_masked = mask * vector
        shifted = mask * (input_masked - K.max(input_masked, axis=1,
                                               keepdims=True))
        # We add epsilon to avoid numerical instability when
        # the sum in the log yields 0.
        normalization_constant = K.log(K.sum(mask * K.exp(shifted), axis=1,
                                             keepdims=True) + K.epsilon())
        normalized_log_probabilities = mask * (shifted - normalization_constant)
        unmasked_probabilities = K.exp(normalized_log_probabilities)
        return switch(mask, unmasked_probabilities, K.zeros_like(unmasked_probabilities))
    else:
        # There is no mask, so we use the provided ``K.softmax`` function.
        return K.softmax(vector)
