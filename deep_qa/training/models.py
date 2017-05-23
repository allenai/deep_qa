import logging
import os
from overrides import overrides

from keras.models import Model, Sequential
import keras.backend as K
import tensorflow

from .step import Step
from ..common.params import Params, ConfigurationError

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
