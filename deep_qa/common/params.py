from typing import Any, Dict, List, Union
from collections import MutableMapping

import logging
import pyhocon

from overrides import overrides
from .checks import ConfigurationError

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

PARAMETER = 60
logging.addLevelName(PARAMETER, "PARAM")


def __param(self, message, *args, **kws):
    """
    Add a method to logger which allows us
    to always log parameters unless you set the logging
    level to be higher than 60 (which is higher than the
    standard highest level of 50, corresponding to WARNING).
    """
    # Logger takes its '*args' as 'args'.
    if self.isEnabledFor(PARAMETER):
        self._log(PARAMETER, message, args, **kws) # pylint: disable=protected-access
logging.Logger.param = __param


class Params(MutableMapping):
    """
    A Class representing a parameter dictionary with a history. Using this allows
    exact reproduction of a parameter file used in an experiment from logs, even when
    default values are used.
    """

    # This allows us to check for the presence of "None" as a default argument,
    # which we require because we make a distinction bewteen passing a value of "None"
    # and passing no value to the default parameter of "pop".
    DEFAULT = object()

    def __init__(self, params: Dict[str, Any], history: str=""):

        self.params = params
        self.history = history

    @overrides
    def pop(self, key: str, default: Any=DEFAULT):
        """
        Performs the functionality associated with dict.pop(key), along with checking for
        returned dictionaries, replacing them with Param objects with an updated history.
        """
        if default is self.DEFAULT:
            value = self.params.pop(key)
        else:
            value = self.params.pop(key, default)
        logger.param(self.history + "." + key + " = " + str(value))
        return self.__check_is_dict(key, value)

    @overrides
    def get(self, key: str, default: Any=DEFAULT):
        """
        Performs the functionality associated with dict.get(key) but also checks for returned
        dicts and returns a Params object in their place with an updated history.
        """
        if default is self.DEFAULT:
            value = self.params.get(key)
        else:
            value = self.params.get(key, default)
        return self.__check_is_dict(key, value)

    def pop_choice(self, key: str, choices: List[Any]):
        """
        Gets the value of ``key`` in the ``params`` dictionary, ensuring that the value is one of the given
        choices. Note that this `pops` the key from params, modifying the dictionary, consistent with how
        parameters are processed in this codebase.
        """

        value = self.pop(key)
        if value not in choices:
            raise ConfigurationError(_get_choice_error_message(value, choices, self.history))
        return value

    def pop_choice_with_default(self,
                                key: str,
                                choices: List[Any],
                                default: Any=None):
        """
        Like pop_choice, but with a default value.  If ``default` is None, we use the first item in
        ``choices`` as the default.
        """
        try:
            return self.pop_choice(key, choices)
        except KeyError:
            if default is None:
                default = choices[0]
            if default not in choices:
                raise ConfigurationError(_get_choice_error_message(default, choices, self.history))

            return self.__check_is_dict(key, default)

    def as_dict(self):
        """
        Sometimes we need to just represent the parameters as a dict, for instance when we pass
        them to a Keras layer(so that they can be serialised).
        """
        def log_recursively(parameters, history):
            for key, value in parameters.items():
                if isinstance(value, dict):
                    new_local_history = value if history == "" else history + "." + key
                    log_recursively(value, new_local_history)
                else:
                    logger.param(history + "." + key + " = " + str(value))

        logger.info("Converting Params object to dict; logging of default "
                    "values will not occur when dictionary parameters are "
                    "used subsequently.")
        logger.info("CURRENTLY DEFINED PARAMETERS: ")
        log_recursively(self.params, self.history)
        return self.params

    def __check_is_dict(self, new_history, value):
        if isinstance(value, dict):
            new_history = new_history if self.history == "" else self.history + "." + new_history
            return Params(value, new_history)
        else:
            return value

    def __getitem__(self, key):
        return self.__check_is_dict(key, self.params[key])

    def __setitem__(self, key, value):
        self.params[key] = value

    def __delitem__(self, key):
        del self.params[key]

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)


def _get_choice_error_message(value: Any, choices: List[Any], name: str=None) -> str:
    if name:
        return '%s not in acceptable choices for %s: %s' % (value, name, str(choices))
    else:
        return '%s not in acceptable choices: %s' % (value, str(choices))


def pop_choice_with_default(params: Dict,
                            key: str,
                            choices: List[Any],
                            default: Any=None,
                            name: str=None) -> Any:
    """
    Performs the same function as the :func:`Params.pop_with_default_method` of Params,
    but is required in order to deal with places that the Params object is not
    welcome, such as inside Keras layers.
    """
    try:
        value = params.pop(key)
    except KeyError:
        if default is None:
            value = choices[0]
        else:
            value = default
    if value not in choices:
        raise ConfigurationError(_get_choice_error_message(default, choices, name))

    logger.param("UNSCOPED PARAMETER: Retrieve default arguments manually."
                 + key + " : " + str(value))

    return value


def assert_params_empty(params: Union[Params, Dict], class_name: str):
    """
    Raises a ConfigurationError if ``params`` is not empty, with a message about where the extra
    parameters were passed to.
    """
    if len(params) != 0:
        raise ConfigurationError("Extra parameters passed to {}: {}".format(class_name, params))


def replace_none(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    for key in dictionary.keys():
        if dictionary[key] == "None":
            dictionary[key] = None
        elif isinstance(dictionary[key], pyhocon.config_tree.ConfigTree):
            dictionary[key] = replace_none(dictionary[key])
    return dictionary
