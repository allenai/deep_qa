import os

PYTHONHASHSEED = '2157'

class ConfigurationError(Exception):
    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


def ensure_pythonhashseed_set():
    message = """You must set PYTHONHASHSEED so that we get repeatable results.
    You can do this with the command `export PYTHONHASHSEED=%s`.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED for more info.
    """
    assert os.environ.get('PYTHONHASHSEED', None) == PYTHONHASHSEED, message % PYTHONHASHSEED
