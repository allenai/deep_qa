import logging
import os
from typing import List
from overrides import overrides

from keras.models import Model, Sequential
from keras.engine.training import _batch_shuffle, _make_batches, _slice_arrays
from keras.callbacks import History, CallbackList, ProgbarLogger, BaseLogger, Callback
import keras.backend as K
import tensorflow
import numpy

from .step import Step
from ..common.params import Params, ConfigurationError
from .train_utils import slice_batch

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class DeepQaModel(Model):
    """
    This is a Model that adds functionality to Keras' ``Model`` class. In
    particular, we use tensorflow optimisers directly in order to make use
    of sparse gradient updates, which Keras does not handle. Additionally,
    we provide some nicer summary functions which include mask information.
    We are overriding key components of Keras here and you should probably
    have a pretty good grip on the internals of Keras before you change
    stuff below, as there could be unexpected consequences.
    """

    # TODO(Mark): Tensorflow optimisers are not compatible with Keras' LearningRateScheduler.
    def __init__(self, *args, **kwargs):
        super(DeepQaModel, self).__init__(*args, **kwargs)

    # We want to add a few things to the summary that's printed out by Keras.  Unfortunately, Keras
    # makes that very difficult.  We have to copy large portions of code in order to make this
    # work, because `print_summary()` is in `keras.utils.layer_utils`, instead of a member on
    # `Container`...
    @overrides
    def summary(self, show_masks=False, **kwargs):
        if show_masks:
            self._summary_with_mask_info()
        else:
            self._keras_summary(**kwargs)

    def _keras_summary(self):
        super(DeepQaModel, self).summary()

    def _summary_with_mask_info(self):
        flattened_layers = getattr(self, 'flattened_layers', self.layers)
        print_summary_with_masking(flattened_layers, getattr(self, 'container_nodes', None))

    @overrides
    def compile(self, params: Params):  # pylint: disable=arguments-differ
        # pylint: disable=attribute-defined-outside-init
        """
        The only reason we are overriding this method is because keras automatically wraps
        our tensorflow optimiser in a keras wrapper, which we don't want. We override the
        only method in ``Model`` which uses this attribute, ``_make_train_function``, which
        raises an error if compile is not called first.
        As we move towards using a Tensorflow first optimisation loop, more things will be
        added here which add functionality to the way Keras runs tensorflow Session calls.

        """
        optimizer = params.get('optimizer')
        self.num_gpus = params.pop('num_gpus', 0)
        self.tensorboard_log = params.pop('tensorboard_log', None)
        self.tensorboard_frequency = params.pop('tensorboard_frequency', 0)
        self.gradient_clipping = params.pop("gradient_clipping", None).as_dict()
        super(DeepQaModel, self).compile(**params.as_dict())
        self.optimizer = optimizer

    @overrides
    def _make_train_function(self):
        # pylint: disable=attribute-defined-outside-init
        """
        We override this method so that we can use tensorflow optimisers directly.
        This is desirable as tensorflow handles gradients of sparse tensors efficiently.
        """
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.train_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]

            tensorflow.summary.scalar("total_loss", self.total_loss)
            # Here we override Keras to use tensorflow optimizers directly.
            self.global_step = tensorflow.train.get_or_create_global_step()
            gradients = tensorflow.gradients(self.total_loss, self._collected_trainable_weights)
            if self.gradient_clipping is not None:
                # Don't pop from the gradient clipping dict here as
                # if we call fit more than once we need it to still be there.
                clip_type = self.gradient_clipping.get("type")
                clip_value = self.gradient_clipping.get("value")
                if clip_type == 'clip_by_norm':
                    gradients, _ = tensorflow.clip_by_global_norm(gradients, clip_value)
                elif clip_type == 'clip_by_value':
                    gradients = [tensorflow.clip_by_value(x, -clip_value, clip_value) for x in gradients]
                else:
                    raise ConfigurationError("{} is not a supported type of gradient clipping.".format(clip_type))

            zipped_grads_with_weights = zip(gradients, self._collected_trainable_weights)
            # pylint: disable=no-member
            training_updates = self.optimizer.apply_gradients(zipped_grads_with_weights,
                                                              global_step=self.global_step)
            # pylint: enable=no-member
            updates = self.updates + [training_updates]
            outputs = [self.total_loss] + self.metrics_tensors
            # Gets loss and metrics. Updates weights at each call.

            if self.tensorboard_log is not None:
                train_summary_writer = tensorflow.summary.FileWriter(os.path.join(self.tensorboard_log, "train"))
            else:
                train_summary_writer = None

            self.train_function = Step(inputs, outputs, self.global_step, train_summary_writer,
                                       self.tensorboard_frequency, updates=updates)

    @overrides
    def _make_test_function(self):
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, 'test_function'):
            raise RuntimeError('You must compile your model before using it.')
        if self.test_function is None:
            inputs = self._feed_inputs + self._feed_targets + self._feed_sample_weights
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs += [K.learning_phase()]
            # Return loss and metrics, no gradient updates.
            # Does update the network states.

            if not hasattr(self, 'global_step'):
                self.global_step = tensorflow.train.get_or_create_global_step()
            self.test_function = Step(inputs, [self.total_loss] + self.metrics_tensors,
                                      self.global_step, updates=self.state_updates)

    @overrides
    def _make_predict_function(self):
        # pylint: disable=attribute-defined-outside-init
        if not hasattr(self, 'predict_function'):
            self.predict_function = None
        if self.predict_function is None:
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                inputs = self._feed_inputs + [K.learning_phase()]
            else:
                inputs = self._feed_inputs
            # Gets network outputs. Does not update weights.
            # Does update the network states.

            if not hasattr(self, 'global_step'):
                self.global_step = tensorflow.train.get_or_create_global_step()
            self.predict_function = Step(inputs, self.outputs, self.global_step,
                                         updates=self.state_updates)

    @overrides
    def _fit_loop(self,
                  f: callable,
                  ins: List[numpy.array],
                  out_labels: List[str]=None,
                  batch_size: int=32,
                  epochs: int=100,
                  verbose: int=1,
                  callbacks: List[Callback]=None,
                  val_f: callable=None,
                  val_ins: List[numpy.array]=None,
                  shuffle: bool=True,
                  callback_metrics: List[str]=None,
                  initial_epoch: int=0):
        """
        Abstract fit function which preprocesses and batches
        data before training a model. We override this keras backend
        function to support multi-gpu training via splitting a large
        batch size across multiple gpus. This function is broadly the
        same as the Keras backend version aside from this - changed elements
        have corresponding comments attached.

        Note that this should not be called directly - it is used by calling
        model.fit().

        Assume that step_function returns a list, labeled by out_labels.

        Parameters
        ----------
        f: A callable ``Step`` or a Keras ``Function``, required.
            A DeepQA Step or Keras Function returning a list of tensors.
        ins: List[numpy.array], required.
            The list of tensors to be fed to ``step_function``.
        out_labels: List[str], optional (default = None).
            The display names of the outputs of ``step_function``.
        batch_size: int, optional (default = 32).
            The integer batch size.
        epochs: int, optional (default = 100).
            Number of times to iterate over the data.
        verbose: int, optional, (default = 1)
            Verbosity mode, 0, 1 or 2.
        callbacks: List[Callback], optional (default = None).
            A list of Keras callbacks to be called during training.
        val_f: A callable ``Step`` or a Keras ``Function``, optional (default = None).
            The Keras function to call for validation.
        val_ins: List[numpy.array], optional (default)
            A list of tensors to be fed to ``val_f``.
        shuffle: bool, optional (default = True).
            whether to shuffle the data at the beginning of each epoch
        callback_metrics: List[str], optional, (default = None).
            A list of strings, the display names of the validation metrics.
            passed to the callbacks. They should be the concatenation of list the display
            names of the outputs of ``f`` and the list of display names of the outputs of ``f_val``.
        initial_epoch: int, optional (default = 0).
            The epoch at which to start training (useful for resuming a previous training run).

        Returns
        -------
        A Keras ``History`` object.

        """
        do_validation = False
        if val_f and val_ins:
            do_validation = True
            if verbose:
                print('Train on %d samples, validate on %d samples' %
                      (ins[0].shape[0], val_ins[0].shape[0]))

        if ins and hasattr(ins[0], 'shape'):
            num_train_samples = ins[0].shape[0]
        else:
            # May happen if we are running `fit` without Numpy input data,
            # i.e. if all inputs to the models are data tensors
            # instead of placeholders.
            # In that case we will run `fit` over a single batch.
            num_train_samples = batch_size
            verbose = 2
        index_array = numpy.arange(num_train_samples)
        out_labels = out_labels or []
        callbacks, callback_model = self._prepare_callbacks(callbacks, val_ins, epochs, batch_size,
                                                            num_train_samples, callback_metrics,
                                                            do_validation, verbose)

        for epoch in range(initial_epoch, epochs):
            callbacks.on_epoch_begin(epoch)
            if shuffle == 'batch':
                index_array = _batch_shuffle(index_array, batch_size)
            elif shuffle:
                numpy.random.shuffle(index_array)

            batches = _make_batches(num_train_samples, batch_size)
            epoch_logs = {}
            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                try:
                    if isinstance(ins[-1], float):
                        # Do not slice the training phase flag.
                        ins_batch = _slice_arrays(ins[:-1], batch_ids) + [ins[-1]]
                    else:
                        ins_batch = _slice_arrays(ins, batch_ids)
                except TypeError:
                    raise TypeError('TypeError while preparing batch. '
                                    'If using HDF5 input data, '
                                    'pass shuffle="batch".')

                # Here is the main difference between a single gpu model and one split
                # across multiple gpus. In our multiple gpu model, all of the inputs
                # are replicated num_gpus times, so we need to split our large batch
                # into the corresponding sets of smaller batches for each model.
                if self.num_gpus > 1:

                    # The Keras learning phase is a global variable used across model towers.
                    # If it is present, we remove it before splitting up the inputs
                    # and add it back on afterwards.
                    if isinstance(ins_batch[-1], float):
                        model_inputs = self._multi_gpu_batch(ins_batch[:-1])
                        model_inputs.append(ins_batch[-1])
                    else:
                        model_inputs = self._multi_gpu_batch(ins_batch)
                    ins_batch = model_inputs

                batch_logs = {}
                batch_logs['batch'] = batch_index
                batch_logs['size'] = len(batch_ids)
                callbacks.on_batch_begin(batch_index, batch_logs)
                outs = f(ins_batch)
                if not isinstance(outs, list):
                    outs = [outs]
                for label, output in zip(out_labels, outs):
                    batch_logs[label] = output

                callbacks.on_batch_end(batch_index, batch_logs)

                if batch_index == len(batches) - 1:  # Last batch.
                    if do_validation:
                        # If we are using multiple gpus, our batch size will be
                        # scaled up accordingly. However, validation will run
                        # on a single gpu, so we divide by the number of gpus
                        # to avoid OOM errors.
                        if self.num_gpus > 1:
                            val_batch_size = int(batch_size/self.num_gpus)  # pylint: disable=no-member
                        else:
                            val_batch_size = batch_size

                        val_outs = self._test_loop(val_f, val_ins,
                                                   batch_size=val_batch_size,
                                                   verbose=0)
                        if not isinstance(val_outs, list):
                            val_outs = [val_outs]
                        # Same labels assumed.
                        for label, output in zip(out_labels, val_outs):
                            epoch_logs['val_' + label] = output
            callbacks.on_epoch_end(epoch, epoch_logs)
            if callback_model.stop_training:  # pylint: disable=no-member
                break
        callbacks.on_train_end()
        return self.history

    def _multi_gpu_batch(self, variable_list):
        # Splits up and orders a list of inputs for a single
        # model into a single list of inputs for that model
        # in a towered fashion, with each input split across the batch size.
        split_batch = slice_batch(variable_list, self.num_gpus)  # pylint: disable=no-member
        ordered_var_list = []
        for single_model_variables in zip(*split_batch):
            ordered_var_list.extend(single_model_variables)
        return ordered_var_list

    def _prepare_callbacks(self,
                           callbacks: List[Callback],
                           val_ins: List[numpy.array],
                           epochs: int,
                           batch_size: int,
                           num_train_samples: int,
                           callback_metrics: List[str],
                           do_validation: bool,
                           verbose: int):

        """
        Sets up Keras callbacks to perform various monitoring functions during training.
        """

        self.history = History()  # pylint: disable=attribute-defined-outside-init
        callbacks = [BaseLogger()] + (callbacks or []) + [self.history]
        if verbose:
            callbacks += [ProgbarLogger()]
        callbacks = CallbackList(callbacks)

        # it's possible to callback a different model than self
        # (used by Sequential models).
        if hasattr(self, 'callback_model') and self.callback_model:
            callback_model = self.callback_model
        else:
            callback_model = self  # pylint: disable=redefined-variable-type

        callbacks.set_model(callback_model)
        callbacks.set_params({
                'batch_size': batch_size,
                'epochs': epochs,
                'samples': num_train_samples,
                'verbose': verbose,
                'do_validation': do_validation,
                'metrics': callback_metrics or [],
        })
        callbacks.on_train_begin()
        callback_model.stop_training = False
        for cbk in callbacks:
            cbk.validation_data = val_ins

        return callbacks, callback_model


def print_summary_with_masking(layers, relevant_nodes=None):
    line_length = 150
    positions = [40, 60, 68, 98, 124, 150]
    headers = ['Layer (type)', 'Output Shape', 'Param #', 'Connected to', 'Input mask', 'Output mask']

    print('_' * line_length)
    print_row(headers, positions)
    print('=' * line_length)

    for i, layer in enumerate(layers):
        print_layer_summary(layer, relevant_nodes, positions)
        if i == len(layers) - 1:
            print('=' * line_length)
        else:
            print('_' * line_length)

    print('Total params: %s' % count_total_params(layers))
    print('_' * line_length)


def print_row(fields, positions):
    line = ''
    for field, position in zip(fields, positions):
        line += str(field)
        line = line[:position - 1]
        line += ' ' * (position - len(line))
    print(line)


def print_layer_summary(layer, relevant_nodes, positions):
    try:
        output_shape = layer.output_shape
    except Exception:  # pylint: disable=broad-except
        output_shape = 'multiple'
    connections = []
    input_masks = []
    output_masks = []
    for node_index, node in enumerate(layer.inbound_nodes):
        input_mask = layer.get_input_mask_at(node_index)
        if isinstance(input_mask, list):
            input_masks.extend(input_mask)
        else:
            input_masks.append(input_mask)
        output_masks.append(layer.get_output_mask_at(node_index))
        if relevant_nodes:
            node_key = layer.name + '_ib-' + str(node_index)
            if node_key not in relevant_nodes:
                # node is node part of the current network
                continue
        for i in range(len(node.inbound_layers)):
            inbound_layer = node.inbound_layers[i].name
            inbound_node_index = str(node.node_indices[i])
            inbound_tensor_index = str(node.tensor_indices[i])
            connections.append(inbound_layer + '[' + inbound_node_index + '][' + inbound_tensor_index + ']')

    name = layer.name
    cls_name = layer.__class__.__name__
    first_connection = '' if not connections else connections[0]
    first_input_mask = '' if not input_masks else input_masks[0]
    first_output_mask = '' if not output_masks else output_masks[0]
    fields = [
            name + ' (' + cls_name + ')',
            output_shape,
            layer.count_params(),
            first_connection,
            first_input_mask,
            first_output_mask,
            ]
    print_row(fields, positions)
    rows_needed = max(len(connections), len(output_masks), len(input_masks))
    for i in range(1, rows_needed):
        connection = '' if i >= len(connections) else connections[i]
        input_mask = '' if i >= len(input_masks) else input_masks[i]
        output_mask = '' if i >= len(output_masks) else output_masks[i]
        fields = ['', '', '', connection, input_mask, output_mask]
        print_row(fields, positions)


def count_total_params(layers, layer_set=None):
    if layer_set is None:
        layer_set = set()
    total_params = 0
    for layer in layers:
        if layer in layer_set:
            continue
        layer_set.add(layer)
        if isinstance(layer, Model) or isinstance(layer, Sequential):
            total_params += count_total_params(layer.layers, layer_set)
        else:
            total_params += layer.count_params()
    return total_params
