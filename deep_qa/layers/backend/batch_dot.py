import keras.backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer


class BatchDot(MaskedLayer):
    """
    This ``Layer`` calls ``K.batch_dot()`` on two inputs ``tensor_a`` and ``tensor_b``.
    This function will work for tensors of arbitrary size as long as
    ``abs(K.ndim(tensor_a) - K.ndim(tensor_b)) < 1``, due to limitations in ``K.batch_dot()``.
    When the input tensors have more than three dimensions, they must have the same shape, except
    for the last two dimensions. See the examples for more explanation of what this means.

    We always assume the dimension to perform the dot is the last one, and that
    the masks have one fewer dimension that the tensors. Note that this layer
    does not return zeroes in places that are masked, but does pass a correct
    mask forward. If this then gets fed into ``masked_softmax``, for instance,
    your tensor will be correctly normalized. We always assume the dimension to
    perform the dot is the last one, and that the masks have one fewer
    dimension than the tensors.

    Inputs:
        - tensor_a: tensor with ``ndim >= 2``.
        - tensor_b: tensor with ``ndim >= 2``.

    Output:
        - a_dot_b

    Examples
    --------
    The following examples will try to give some insight on how this layer works in relation
    to ``K.batch_dot()``. Note that the Keras documentation (as of 2/13/17) on ``K.batch_dot``
    is incorrect, and that this layer behaves differently from the documented behavior.

    As a first example, let's suppose that ``tensor_a`` and ``tensor_b`` have the same
    number of dimensions. Let the shape of ``tensor_a`` be ``(2, 3, 2)``, and let the shape
    of ``tensor_b`` be ``(2, 4, 2)``. The mask accompanying these inputs always has one less
    dimension, so the ``tensor_a_mask`` has shape ``(2, 3)`` and ``tensor_b_mask`` has
    shape ``(2, 4)``. The shape of the ``batch_dot`` output would thus be ``(2, 3, 4)``. This is
    because we are taking the batch dot of the last dimension, so the output shape is ``(2, 3)``
    (from tensor_a) with ``(4)`` (from tensor_b) appended on (to get ``(2, 3, 4)`` in total). The
    output mask has the same shape as the output, and is thus ``(2, 3, 4)`` as well.

    >>> import keras.backend as K
    >>> tensor_a = K.ones(shape=(2, 3, 2))
    >>> tensor_b = K.ones(shape=(2, 4, 2))
    >>> K.eval(K.batch_dot(tensor_a, tensor_b, axes=(2,2))).shape
    (2, 3, 4)

    Next, let's look at an example where ``tensor_a`` and ``tensor_b`` are "uneven" (different number
    of dimensions). Let the shape of ``tensor_a`` be ``(2, 4, 2)``, and let the shape of ``tensor_b``
    be ``(2, 4, 3, 2)``. The mask accompanying these inputs always has one less dimension, so the
    ``tensor_a_mask`` has shape ``(2, 4)`` and ``tensor_b_mask`` has shape ``(2, 4, 3)``. The shape
    of the ``batch_dot`` output would thus be ``(2, 4, 3)``. In the case of uneven tensors, we always
    expand the last dimension of the smaller tensor to make them even. Thus in this case, we expand
    ``tensor_a`` to get a new shape of ``(2, 4, 2, 1)``. Now we are taking the ``batch_dot`` of a
    tensor with shape ``(2, 4, 2, 1)`` and ``(2, 4, 3, 2)``. Note that the first two dimensions of
    this tensor are the same ``(2, 4)`` -- this is a requirement imposed by ``K.batch_dot``.
    Following the methodology of calculating the output shape above, we get that the output is
    ``(2, 4, 1, 3)`` since we get ``(2, 4, 1)`` from ``tensor_a`` and ``(3)`` from ``tensor_b``. We
    then squeeze the tensor to remove the 1-dimension to get a final shape of ``(2, 4, 3)``. Note
    that the mask has the same shape.

    >>> import keras.backend as K
    >>> tensor_a = K.ones(shape=(2, 4, 2))
    >>> tensor_b = K.ones(shape=(2, 4, 3, 2))
    >>> tensor_a_expanded = K.expand_dims(tensor_a, axis=-1)
    >>> unsqueezed_bd = K.batch_dot(tensor_a_expanded, tensor_b, axes=(2,3))
    >>> final_bd = K.squeeze(unsqueezed_bd, axis=K.ndim(tensor_a)-1)
    >>> K.eval(final_bd).shape
    (2, 4, 3)

    Lastly, let's look at the uneven case where ``tensor_a`` has more dimensions than ``tensor_b``.
    Let the shape of ``tensor_a`` be ``(2, 3, 4, 2)``, and let the shape of ``tensor_b``
    be ``(2, 3, 2)``. Since the mask accompanying these inputs always has one less dimension,
    ``tensor_a_mask`` has shape ``(2, 3, 4)`` and ``tensor_b_mask`` has shape ``(2, 3)``. The shape
    of the ``batch_dot`` output would thus be ``(2, 3, 4)``. Since these tensors are uneven, expand
    the smaller tensor, ``tensor_b``, to get a new shape of ``(2, 3, 2, 1)``. Now we are taking
    the ``batch_dot`` of a tensor with shape ``(2, 3, 4, 2)`` and ``(2, 3, 2, 1)``. Note again that the
    first two dimensions of this tensor are the same ``(2, 3)``. We can see that the output shape is
    ``(2, 3, 4, 1)`` since we get ``(2, 3, 4)`` from ``tensor_a`` and ``(1)`` from ``tensor_b``. We
    then squeeze the tensor to remove the 1-dimension to get a final shape of ``(2, 3, 4)``. Note
    that the mask has the same shape.

    >>> import keras.backend as K
    >>> tensor_a = K.ones(shape=(2, 3, 4, 2))
    >>> tensor_b = K.ones(shape=(2, 3, 2))
    >>> tensor_b_expanded = K.expand_dims(tensor_b, axis=-1)
    >>> unsqueezed_bd = K.batch_dot(tensor_a, tensor_b_expanded, axes=(3, 2))
    >>> final_bd = K.squeeze(unsqueezed_bd, axis=K.ndim(tensor_a)-1)
    >>> K.eval(final_bd).shape
    (2, 3, 4)

    """
    def __init__(self, **kwargs):
        super(BatchDot, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        tensor_a, tensor_b = inputs
        mask_a, mask_b = mask
        a_dot_axis = K.ndim(tensor_a) - 1
        b_dot_axis = K.ndim(tensor_b) - 1

        if mask_a is None and mask_b is None:
            return None
        elif mask_a is None:
            mask_a = K.sum(K.ones_like(tensor_a), axis=-1)
        elif mask_b is None:
            # (batch_size, b_length)
            mask_b = K.sum(K.ones_like(tensor_b), axis=-1)
        float_mask_a = K.cast(mask_a, "float32")
        float_mask_b = K.cast(mask_b, "float32")
        if b_dot_axis == a_dot_axis:
            # tensor_a and tensor_b have the same length.
            float_mask_a = K.expand_dims(float_mask_a, axis=-1)
            float_mask_b = K.expand_dims(float_mask_b, axis=-1)
            final_mask = K.batch_dot(float_mask_a, float_mask_b,
                                     axes=(a_dot_axis, b_dot_axis))
        elif a_dot_axis < b_dot_axis:
            # tensor_a has less dimensions than tensor_b.
            # We would tile tensor_a to have the same shape as tensor_b,
            # but we can just expand tensor_a and have TF broadcast
            # over the last dimension
            float_mask_a = K.expand_dims(float_mask_a, axis=-1)
            final_mask = float_mask_a * float_mask_b
        else:
            # tensor_a has more dimensions than tensor_b.
            # We would tile tensor_b to have the same shape as tensor_a,
            # but we can just expand tensor_b and have TF broadcast
            # over the last dimension
            float_mask_b = K.expand_dims(float_mask_b, axis=-1)
            final_mask = float_mask_a * float_mask_b
        return final_mask

    @overrides
    def compute_output_shape(self, input_shape):
        tensor_a_shape, tensor_b_shape = input_shape
        a_dot_axis = len(tensor_a_shape) - 1
        b_dot_axis = len(tensor_b_shape) - 1
        if b_dot_axis < a_dot_axis:
            tensor_b_shape += (1,)
        if b_dot_axis > a_dot_axis:
            tensor_a_shape += (1,)

        # This assumes that we do the dot product over the last dimension
        final_out_shape = []
        for i in range(0, len(tensor_a_shape)):
            if i != a_dot_axis:
                final_out_shape.append(tensor_a_shape[i])
        for i in range(len(tensor_b_shape)-2, len(tensor_b_shape)):
            if i != b_dot_axis and i != 0:
                final_out_shape.append(tensor_b_shape[i])
        if b_dot_axis != a_dot_axis:
            # remove the 1 we inserted
            final_out_shape.pop(a_dot_axis)
        if len(final_out_shape) == 1:
            final_out_shape.append(1)
        return tuple(final_out_shape)

    @overrides
    def call(self, inputs, mask=None):
        tensor_a, tensor_b = inputs
        a_dot_axis = K.ndim(tensor_a) - 1
        b_dot_axis = K.ndim(tensor_b) - 1
        if a_dot_axis > b_dot_axis:
            tensor_b = K.expand_dims(tensor_b, axis=-1)
        if a_dot_axis < b_dot_axis:
            tensor_a = K.expand_dims(tensor_a, axis=-1)
        a_dot_b = K.batch_dot(tensor_a, tensor_b, axes=(a_dot_axis, b_dot_axis))
        if a_dot_axis != b_dot_axis:
            a_dot_b = K.squeeze(a_dot_b, axis=a_dot_axis)
        return a_dot_b
