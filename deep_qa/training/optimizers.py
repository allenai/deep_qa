r"""
It turns out that Keras' design is somewhat crazy\*, and there is no list of
optimizers that you can just import from Keras. So, this module specifies a
list, and a helper function or two for dealing with optimizer parameters.
Unfortunately, this means that we have a list that must be kept in sync with
Keras. Oh well.

\* Have you seen their get_from_module() method? See here:
https://github.com/fchollet/keras/blob/6e42b0e4a77fb171295b541a6ae9a3a4a79f9c87/keras/utils/generic_utils.py#L10.
That method means I could pass in 'clip_norm' as an optimizer, and it would try
to use that function as an optimizer. It also means there is no simple list of
implemented optimizers I can grab.

\* I should also note that Keras is an incredibly useful library that does a lot
of things really well. It just has a few quirks...
"""
from typing import Union
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from ..common.params import Params

optimizers = {  # pylint: disable=invalid-name
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
        }


def optimizer_from_params(params: Union[Params, str]):
    """
    This method converts from a parameter object like we use in our Trainer
    code into an optimizer object suitable for use with Keras. The simplest
    case for both of these is a string that shows up in `optimizers` above - if
    `params` is just one of those strings, we return it, and everyone is happy.
    If not, we assume `params` is a Dict[str, Any], with a "type" key, where
    the value for "type" must be one of those strings above. We take the rest
    of the parameters and pass them to the optimizer's constructor.

    """
    if isinstance(params, str) and params in optimizers.keys():
        return params
    optimizer = params.pop_choice("type", optimizers.keys())
    return optimizers[optimizer](**params)
