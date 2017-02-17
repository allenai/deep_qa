from overrides import overrides

from keras.models import Model, Sequential


class DeepQaModel(Model):
    """
    This is a Model that adds a little bit more functionality than is present in Keras Models.
    """
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
