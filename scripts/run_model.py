import logging
import os
import shutil
import sys

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
import numpy
random.seed(13370)
numpy.random.seed(1337)

# pylint: disable=wrong-import-position

# Unless I find a better way to do this, we'll use ConfigFactory to read in a parameter file.
# ConfigFactory.parse_file() returns a ConfigTree, which is a subclass of OrderedDict, so the
# return value actually matches the type our code expects, anyway.
import pyhocon

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deep_qa.common.checks import ensure_pythonhashseed_set, log_keras_version_info
from deep_qa.common.params import Params, replace_none
from deep_qa.common.tee_logger import TeeLogger
from deep_qa.models import concrete_models
from keras import backend as K

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    if len(sys.argv) != 2:
        print('USAGE: run_model.py [param_file]')
        sys.exit(-1)

    log_keras_version_info()
    param_file = sys.argv[1]
    param_dict = pyhocon.ConfigFactory.parse_file(param_file)
    params = Params(replace_none(param_dict))
    log_dir = params.get("model_serialization_prefix", None)  # pylint: disable=no-member
    if log_dir is not None:
        sys.stdout = TeeLogger(log_dir + "_stdout.log", sys.stdout)
        sys.stderr = TeeLogger(log_dir + "_stderr.log", sys.stderr)
        handler = logging.FileHandler(log_dir + "_python_logging.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        shutil.copyfile(param_file, log_dir + "_model_params.json")
    model_type = params.pop_choice('model_class', concrete_models.keys())
    model_class = concrete_models[model_type]
    model = model_class(params)

    if model.can_train():
        logger.info("Training model")
        model.train()
    else:
        logger.info("Not enough training inputs.  Assuming you wanted to load a model instead.")
        # TODO(matt): figure out a way to specify which epoch you want to load a model from.
        model.load_model()
    K.clear_session()


if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
