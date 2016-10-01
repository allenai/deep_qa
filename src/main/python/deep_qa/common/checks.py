import os

PYTHONHASHSEED = 2157

def ensure_pythonhashseed_set():
    message = """You must set PYTHONHASHSEED so that we get repeatable results.
    You can do this with the command `export PYTHONHASHSEED=%d`.
    See https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED for more info.
    """
    assert os.environ.get('PYTHONHASHSEED', None) == PYTHONHASHSEED, message % PYTHONHASHSEED
