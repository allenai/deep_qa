from typing import Callable
import os
from copy import deepcopy

import tensorflow
import keras.backend as K

from .train_utils import pin_variable_device_scope, average_gradients
from .models import DeepQaModel
from .step import Step
from ..common.params import Params, ConfigurationError


def compile_parallel_model(model_builder: Callable[[], DeepQaModel],
                           compile_arguments: Params) -> DeepQaModel:
    """
    This function compiles a multi-gpu version of your model. This is done using data
    parallelism, by making N copies of the model on the different GPUs, all of which
    share parameters. Gradients are updated synchronously, using the average gradient
    from all of the outputs of the various models. This effectively allows you to scale
    a model up to batch_sizes which cannot fit on a single GPU.

    This method returns a "primary" copy of the model, which has had its training
    function which is run by Keras overridden to be a training function which trains
    all of the towers of the model. The other towers never have their training functions
    initialised or used and are completely hidden from the user. The returned model
    can be serialised in the same way as any other model and has no dependency on
    multiple gpus being available when it is loaded.

    Note that by calling this function, the model_builder function will be called multiple times
    for the different GPUs. As such, you should be wary of this function having side
    effects unrelated to building a computation graph.

    Parameters
    ----------

    model_builder: Callable[any, DeepQaModel], required.
        A function which returns an uncompiled DeepQaModel.
    compile_arguments: Params, required
        Model parameters which are passed to compile. These should be the same as if you
        were building a single GPU model, with the exception of the ``num_gpus`` field.

    Returns
    -------
    The "primary" copy of the DeepQaModel, which holds the training function which
    trains all of the copies of the model.
    """

    optimizer = compile_arguments.get("optimizer")
    num_gpus = compile_arguments.get("num_gpus")
    gradient_clipping = compile_arguments.get("gradient_clipping", None)
    tower_models = []
    tower_gradients = []
    global_step = tensorflow.train.get_or_create_global_step()
    train_loss = tensorflow.get_variable('train_loss', [],
                                         initializer=tensorflow.constant_initializer(0.0),
                                         trainable=False)

    # Place a copy of the model on each GPU, each getting a slice of the batch.
    for gpu_index in range(num_gpus):
        with tensorflow.device(pin_variable_device_scope('/gpu:%d' % gpu_index)):
            with tensorflow.name_scope('tower_%d' % gpu_index):  # pylint: disable=not-context-manager
                # This is a new model object every time.
                model = model_builder()
                compile_kwargs = deepcopy(compile_arguments)
                model.compile(compile_kwargs)
                loss = model.total_loss
                tower_models.append(model)
                grads = optimizer.compute_gradients(loss)
                tower_gradients.append(grads)
                train_loss += loss

    grads_and_variables = average_gradients(tower_gradients)

    gradients, variables = list(zip(*grads_and_variables))
    if gradient_clipping is not None:
        clip_type = gradient_clipping.pop("type")
        clip_value = gradient_clipping.pop("value")
        if clip_type == 'clip_by_norm':
            gradients, _ = tensorflow.clip_by_global_norm(gradients, clip_value)
        elif clip_type == 'clip_by_value':
            gradients = [tensorflow.clip_by_value(x, -clip_value, clip_value) for x in gradients]
        else:
            raise ConfigurationError("{} is not a supported type of gradient clipping.".format(clip_type))

    train_operation = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    train_summary = tensorflow.summary.scalar('train_loss', train_loss/ num_gpus)

    summary_operations = [train_summary]
    # any metrics that keras has collected
    merged_metrics = []
    if tower_models[0].metrics is not None:
        # merge the metrics across GPUs
        for i in range(len(tower_models[0].metrics)):
            name = tower_models[0].metrics[0]
            tensor = tensorflow.reduce_mean([mm.metrics_tensors[i] for mm in tower_models])
            summary_operations.append(tensorflow.summary.scalar(name, tensor))
            merged_metrics.append(tensor)

    inputs = []
    updates = []
    for model in tower_models:
        # pylint: disable=protected-access
        model_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
        # pylint: enable=protected-access
        inputs.extend(model_inputs)
        updates.extend(model.updates)
    # Just check any one, as we just made copies of them.
    if tower_models[0].uses_learning_phase and \
            not isinstance(K.learning_phase(), int):
        inputs += [K.learning_phase()]

    primary_model = tower_models[0]
    if primary_model.tensorboard_log is not None:
        train_summary_writer = tensorflow.summary.FileWriter(os.path.join(primary_model.tensorboard_log, "train"))
    else:
        train_summary_writer = None

    # Add the multi-gpu update operation.
    updates += [train_operation]
    # Gets loss and metrics. Updates weights at each call.
    primary_model.train_function = Step(inputs,
                                        [train_loss] + merged_metrics,
                                        global_step,
                                        summary_writer=train_summary_writer,
                                        summary_frequency=primary_model.tensorboard_frequency,
                                        updates=updates)
    return primary_model
