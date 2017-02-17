from typing import Any, Dict, List

from .checks import ConfigurationError


def _get_choice_error_message(value: Any, choices: List[Any], name: str=None) -> str:
    if name:
        return '%s not in acceptable choices for %s: %s' % (value, name, str(choices))
    else:
        return '%s not in acceptable choices: %s' % (value, str(choices))


def get_choice(params: [Dict, Any], key: str, choices: List[Any], name: str=None):
    """
    Gets the value of `key` in the `params` dictionary, ensuring that the value is one of the given
    choices.  `name` is an optional description for where a configuration error happened, if there
    is one.

    Note that this _pops_ the key from params, modifying the dictionary, consistent with how
    parameters are processed in this codebase.
    """
    value = params.pop(key)
    if value not in choices:
        raise ConfigurationError(_get_choice_error_message(value, choices, name))
    return value


def get_choice_with_default(params: [Dict, Any],
                            key: str,
                            choices: List[Any],
                            default: Any=None,
                            name: str=None):
    """
    Like get_choice, but with a default value.  If `default` is None, we use the first item in
    `choices` as the default.
    """
    try:
        return get_choice(params, key, choices, name)
    except KeyError:
        if default is None:
            return choices[0]
        if default not in choices:
            raise ConfigurationError(_get_choice_error_message(default, choices, name))
        return default
