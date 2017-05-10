from keras import backend as K

from ..tensors.backend import switch, VERY_NEGATIVE_NUMBER, VERY_LARGE_NUMBER

def ranking_loss(y_pred, y_true):
    """
    Using this loss trains the model to give scores to all correct elements in y_true that are
    higher than all scores it gives to incorrect elements in y_true.

    For example, let ``y_true = [0, 0, 1, 1, 0]``, and let ``y_pred = [-1, 1, 2, 0, -2]``.  We will
    find the lowest score assigned to correct elements in ``y_true`` (``0`` in this case), and the
    highest score assigned to incorrect elements in ``y_true`` (``1`` in this case).  We will then
    compute a sigmoided loss given these values: ``-K.sigmoid(0 - 1)`` (we're minimizing the loss,
    so the negative sign in front of the sigmoid means we want the correct element to have a higher
    score than the incorrect element).

    Note that the way we do this uses ``K.max()`` and ``K.min()`` over the elements in ``y_true``,
    which means that if you have a lot of values in here, you'll only get gradients backpropping
    through two of them (the ones on the margin).  This could be an inefficient use of your
    computation time.  Think carefully about the data that you're using with this loss function.

    Because of the way masking works with Keras loss functions, also, you need to be sure that any
    masked elements in ``y_pred`` have very negative values before they get passed into this loss
    function.
    """
    correct_elements = switch(y_true, y_pred, K.ones_like(y_pred) * VERY_LARGE_NUMBER)
    lowest_scoring_correct = K.min(correct_elements, axis=-1)
    incorrect_elements = switch(y_true, K.ones_like(y_pred) * VERY_NEGATIVE_NUMBER, y_pred)
    highest_scoring_incorrect = K.max(incorrect_elements, axis=-1)
    return K.mean(-K.sigmoid(lowest_scoring_correct - highest_scoring_incorrect))


def ranking_loss_with_margin(y_pred, y_true):
    """
    Using this loss trains the model to give scores to all correct elements in y_true that are
    higher than all scores it gives to incorrect elements in y_true, plus a margin.

    For example, let ``y_true = [0, 0, 1, 1, 0]``, and let ``y_pred = [-1, 1, 2, 0, -2]``.  We will
    find the lowest score assigned to correct elements in ``y_true`` (``0`` in this case), and the
    highest score assigned to incorrect elements in ``y_true`` (``1`` in this case).  We will then
    compute a hinge loss given these values: ``K.maximum(0.0, 1 + 1 - 0)``.

    Note that the way we do this uses ``K.max()`` and ``K.min()`` over the elements in ``y_true``,
    which means that if you have a lot of values in here, you'll only get gradients backpropping
    through two of them (the ones on the margin).  This could be an inefficient use of your
    computation time.  Think carefully about the data that you're using with this loss function.

    Because of the way masking works with Keras loss functions, also, you need to be sure that any
    masked elements in ``y_pred`` have very negative values before they get passed into this loss
    function.
    """
    correct_elements = switch(y_true, y_pred, K.ones_like(y_pred) * VERY_LARGE_NUMBER)
    lowest_scoring_correct = K.min(correct_elements, axis=-1)
    incorrect_elements = switch(y_true, K.ones_like(y_pred) * VERY_NEGATIVE_NUMBER, y_pred)
    highest_scoring_incorrect = K.max(incorrect_elements, axis=-1)
    return K.mean(K.maximum(0.0, 1.0 + highest_scoring_incorrect - lowest_scoring_correct))
