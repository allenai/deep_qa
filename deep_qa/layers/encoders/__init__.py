from collections import OrderedDict

from keras.layers import LSTM
from keras.layers.wrappers import Bidirectional
from keras.regularizers import l1_l2

from ...common.params import Params
from .bag_of_words import BOWEncoder
from .convolutional_encoder import CNNEncoder
from .positional_encoder import PositionalEncoder
from .shareable_gru import ShareableGRU as GRU


def set_regularization_params(encoder_type: str, params: Params):
    """
    This method takes regularization parameters that are specified in `params` and converts them
    into Keras regularization objects, modifying `params` to contain the correct keys for the given
    encoder_type.

    Currently, we only allow specifying a consistent regularization across all the weights of a
    layer.
    """
    l2_regularization = params.pop("l2_regularization", None)
    l1_regularization = params.pop("l1_regularization", None)
    regularizer = lambda: l1_l2(l1=l1_regularization, l2=l2_regularization)
    if encoder_type == 'cnn':
        # Regularization with the CNN encoder is complicated, so we'll just pass in the L1 and L2
        # values directly, and let the encoder deal with them.
        params["l1_regularization"] = l1_regularization
        params["l2_regularization"] = l2_regularization
    elif encoder_type == 'lstm':
        params["W_regularizer"] = regularizer()
        params["U_regularizer"] = regularizer()
        params["b_regularizer"] = regularizer()
    elif encoder_type == 'tree_lstm':
        params["W_regularizer"] = regularizer()
        params["U_regularizer"] = regularizer()
        params["V_regularizer"] = regularizer()
        params["b_regularizer"] = regularizer()
    return params


# The first item added here will be used as the default in some cases.
encoders = OrderedDict()  # pylint:  disable=invalid-name
encoders["bow"] = BOWEncoder
encoders["lstm"] = LSTM
encoders["gru"] = GRU
encoders["cnn"] = CNNEncoder
encoders["positional"] = PositionalEncoder
encoders["bi_gru"] = (lambda **params: Bidirectional(GRU(return_sequences=False,
                                                         **params)))

seq2seq_encoders = OrderedDict()  # pylint:  disable=invalid-name
seq2seq_encoders["bi_gru"] = (lambda **params:
                              Bidirectional(GRU(return_sequences=True,
                                                **(params["encoder_params"])),
                                            **(params["wrapper_params"])))
