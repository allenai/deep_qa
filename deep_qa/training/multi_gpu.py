from keras.layers import concatenate
from keras.layers.core import Lambda
import keras.backend as K
import tensorflow

from .models import DeepQaModel


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


def make_parallel(model: DeepQaModel, gpu_count: int) -> DeepQaModel:
    """
    Parameters
    ----------
    model: An instance of a DeepQaModel.
    gpu_count: The number of GPUs to duplicate the model across.

    Output:
    - A new DeepQaModel, consisting of model_duplicates over
        the number of available GPUs.
    """
    # Argument to a Lambda layer which will slice our large batches
    # along the batch dimension and return a given slice.
    def get_slice(data, index, parts):
        # We need to re-import tensorflow here so that Keras
        # can serialise the layer correctly.
        import tensorflow  # pylint: disable=redefined-outer-name,reimported
        is_last_slice = (index == parts - 1)

        shape = K.shape(data)
        batch_shape, feature_shape = shape[:1], shape[1:]
        stride = K.concatenate([batch_shape // parts, feature_shape * 0], axis=0)

        if not is_last_slice:
            size = K.concatenate([batch_shape // parts, feature_shape], axis=0)
        else:
            # For the last device we take everything.
            # This deals with the case that big_batch_size % num_gpus != 0.
            size = K.concatenate([[-1], feature_shape], axis=0)
        start = stride * index
        return tensorflow.slice(data, start, size)

    all_outputs = [[] for _ in model.outputs]
    # Place a copy of the model on each GPU, each getting a slice of the batch.
    for gpu_index in range(gpu_count):
        with tensorflow.device(pin_variable_device_scope('/gpu:%d' % gpu_index)):
            with tensorflow.name_scope('tower_%d' % gpu_index):
                inputs = []
                # Slice each input into a piece for processing on this GPU.
                for model_input in model.inputs:
                    # Get the shape of everything apart from the batch,
                    # which will be split across the GPUs.
                    output_shape = tuple(model_input.get_shape().as_list())[1:]
                    slice_layer = Lambda(get_slice,
                                         output_shape=output_shape,
                                         arguments={'index': gpu_index, 'parts': gpu_count})
                    slice_n = slice_layer(model_input)
                    inputs.append(slice_n)

                outputs = model(inputs)
                if not isinstance(outputs, list):
                    outputs = [outputs]

                # Save all the outputs for merging back together later.
                for i, output in enumerate(outputs):
                    all_outputs[i].append(output)

    # Merge outputs on CPU.
    with tensorflow.device('/cpu:0'):
        merged = []
        for outputs in all_outputs:
            merged.append(concatenate(outputs, axis=0))
        return DeepQaModel(input=model.inputs, output=merged)
