from typing import List, Tuple
from collections import defaultdict
import tensorflow


def pin_variable_device_scope(device, variable_device="/cpu:0"):
    """
    Tensorflow device scopes can take functions which give a device
    for a given op in the graph. Here, we use the device that is
    passed to the scope *unless* the operation which is being created
    in the graph is a Variable creation op; in this case, we place it
    on the cpu.
    """
    def _assign(graph_op):
        node_def = graph_op if isinstance(graph_op, tensorflow.NodeDef) else graph_op.node_def
        if node_def.op in ['Variable', 'VariableV2']:
            return variable_device
        else:
            return device
    return _assign


def average_gradients(tower_gradients: List[List[Tuple[tensorflow.Tensor, tensorflow.Tensor]]]):
    """
    Given a list of (gradient, variable) pairs from the result of
    a gradient calculation from multiple GPUs, calculate their
    average.
    """
    # Make a map from variables -> [gradients that are not none].
    gradient_map = defaultdict(list)
    for tower in tower_gradients:
        for grad, variable in tower:
            if grad is not None:
                gradient_map[variable].append(grad)

    average_gradient_list = []
    for variable, gradients in gradient_map.items():
        # variable is a tensor.
        # gradients is a list of gradients for this tensor to average.
        # Pick any one of the gradients to see if it is an IndexedSlice.
        first_actual_grad = gradients[0]
        if isinstance(first_actual_grad, tensorflow.IndexedSlices):
            sparse_averaged_gradient = _get_sparse_gradient_average(gradients)
            average_gradient_list.append((sparse_averaged_gradient, variable))
        else:
            dense_averaged_gradient = _get_dense_gradient_average(gradients)
            average_gradient_list.append((dense_averaged_gradient, variable))
    assert len(average_gradient_list) == len(gradient_map)
    return average_gradient_list


def _get_dense_gradient_average(gradients: List[tensorflow.Tensor]):
    """
    A normal tensor can just do a simple average. Here, we stack all the gradients into a
    tensor and then average over the dimension which they were stacked into.

    Parameters
    ----------
    gradients: List[tensorflow.Tensor])
        The list of gradients to average.

    Returns
    -------
    An average gradient.
    """
    grads_expanded = []
    for grad in gradients:
        # Add a 0 dimension to the gradients to represent the tower and
        # append on a 'tower' dimension which we will average over.
        grads_expanded.append(tensorflow.expand_dims(grad, 0))

    # Average over the 'tower' dimension.
    grad = tensorflow.concat(grads_expanded, 0)
    mean_grad = tensorflow.reduce_mean(grad, 0)

    return mean_grad


def _get_sparse_gradient_average(gradients: List[tensorflow.IndexedSlices]):
    """
    If the gradient is an instance of an IndexedSlices then this is a sparse
    gradient with attributes indices and values. To average, we
    need to concat them individually and then create a new
    IndexedSlices object. This case frequently occurs in the embedding layers
    of neural network models, as for a given input, only some indices of the
    embedding are updated, so performing sparse updates using IndexedSlices
    is considerably more efficient.

    Parameters
    ----------
    gradients: List[tensorflow.IndexedSlices])
        The list of sparse gradients to average.

    Returns
    -------
    An average gradient.

    """
    indices = []
    values = []
    first_actual_gradient = gradients[0]
    for grad in gradients:
        indices.append(grad.indices)
        values.append(grad.values)
    all_indices = tensorflow.concat(indices, 0)
    avg_values = tensorflow.concat(values, 0) / len(gradients)

    # NOTE(Mark): tf.unique has no GPU implementation in tensorflow,
    # so if you use a network which requires sparse gradients for an op which
    # occurs on the GPU (such as tf.gather, tf.scatter), this will be slow.
    # This is not a problem for the embedding lookup, because this already happens
    # on the CPU. See this issue:
    # https://github.com/tensorflow/tensorflow/issues/10270

    # Deduplicate across indices.
    unique_indices, new_index_positions = tensorflow.unique(all_indices)
    deduplicated_values = tensorflow.unsorted_segment_sum(
            avg_values, new_index_positions,
            tensorflow.shape(unique_indices)[0])

    mean_grad = tensorflow.IndexedSlices(deduplicated_values,
                                         unique_indices,
                                         dense_shape=first_actual_gradient.dense_shape)
    return mean_grad


def slice_batch(batch_inputs: List[tensorflow.Tensor], num_gpus: int):
    """
    Given a list of Tensor inputs to a model, split each input into a list of
    tensors of length num_gpus, where the first dimension of each element is
    equal to the original dimension divided by the number of gpus.

    Parameters
    ----------
    batch_inputs: List[tensorflow.Tensor])
        The list of model inputs to split up.
    num_gpus: int
        The number of gpus to split the inputs across.

    Returns
    -------
    all_slices: List[List[tensorflow.Tensor]]
        A list of lists of tensors split across their first dimension by num_gpus.
    """

    all_slices = []
    for placeholder in batch_inputs:
        # splice placeholder into batches split across the number of gpus specified.
        batch_size = int(int(placeholder.shape[0]) / num_gpus)
        placeholder_slices = []
        for i in range(num_gpus):
            placeholder_slices.append(placeholder[(i * batch_size):((i + 1) * batch_size), ...])
        all_slices.append(placeholder_slices)
    return all_slices
