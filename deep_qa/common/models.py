from typing import List

from keras.models import Model

from ..training.models import DeepQaModel


def get_submodel(model: Model,
                 input_layer_names: List[str],
                 output_layer_names: List[str],
                 train_model: bool=False,
                 name=None):
    """
    Returns a new model constructed from ``model``.  This model will be a subset of the given
    ``Model``, with the inputs specified by ``input_layer_names`` and the outputs specified by
    ``output_layer_names``.  For example, if the input model is :class:`BiDAF
    .models.reading_comprehens.bidirectional_attention.BidirectionalAttentionFlow`, you can use
    this to get a model that outputs the passage embedding, just before the span prediction
    layers, by calling
    ``get_submodel(bidaf.model, ['question_input', 'passage_input'], ['final_merged_passage'])``.
    """
    layer_input_dict = {}
    layer_output_dict = {}
    for layer in model.layers:
        layer_input_dict[layer.name] = layer.get_input_at(0)
        layer_output_dict[layer.name] = layer.get_output_at(0)
    input_layers = [layer_input_dict[name] for name in input_layer_names]
    output_layers = [layer_output_dict[name] for name in output_layer_names]
    submodel = DeepQaModel(inputs=input_layers, outputs=output_layers, name=name)
    if not train_model:
        submodel.trainable = False
    return submodel
