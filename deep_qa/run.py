from typing import Union
import sys
import logging
import shutil

import random
import pyhocon
import numpy

# pylint: disable=wrong-import-position
from .common.params import Params, replace_none, ConfigurationError
from .common.tee_logger import TeeLogger

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def prepare_environment(params: Union[Params, dict]):
    """
    Sets random seeds for reproducible experiments. This may not work as expected
    if you use this from within a python project in which you have already imported Keras.
    If you use the scripts/run_model.py entry point to training models with this library,
    your experiments should be reproducible. If you are using this from your own project,
    you will want to call this function before importing Keras.

     Parameters
    ----------
    params: ``Params`` object or dict, required.
        A ``Params`` object or dict holding the json parameters.
    """
    seed = params.pop("random_seed", 13370)
    numpy_seed = params.pop("numpy_seed", 1337)
    if "keras" in sys.modules:
        logger.warning("You have already imported Keras in your code. If you are using DeepQA"
                       "functionality to set random seeds, they will have no effect, as code"
                       "prior to calling this function will be non-deterministic. We will not"
                       "the random seed here.")
        seed = None
        numpy_seed = None
    if seed is not None:
        random.seed(seed)
    if numpy_seed is not None:
        numpy.random.seed(numpy_seed)

    from deep_qa.common.checks import log_keras_version_info
    log_keras_version_info()


def run_model(param_path: str, model_class=None):
    """
    This function is the normal entry point to DeepQA. Use this to run a DeepQA model in
    your project. Note that if you care about exactly reproducible experiments,
    you should avoid importing Keras before you import and use this function, as
    Keras relies on random seeds which can be set in this function via a
    JSON specification file.

    Note that this function performs training and will also evaluate the trained
    model on development and test sets if provided in the parameter json.

    Parameters
    ----------
    param_path: str, required.
        A json file specifying a DeepQaModel.
    model_class: ``DeepQaModel``, optional (default=None).
        This option is useful if you have implemented a new model class which
        is not one of the ones implemented in this library.
    """
    param_dict = pyhocon.ConfigFactory.parse_file(param_path)
    params = Params(replace_none(param_dict))
    prepare_environment(params)

    # These have to be imported _after_ we set the random seed,
    # because keras uses the numpy random seed.
    from deep_qa.models import concrete_models
    from keras import backend as K

    log_dir = params.get("model_serialization_prefix", None)  # pylint: disable=no-member
    if log_dir is not None:
        sys.stdout = TeeLogger(log_dir + "_stdout.log", sys.stdout)
        sys.stderr = TeeLogger(log_dir + "_stderr.log", sys.stderr)
        handler = logging.FileHandler(log_dir + "_python_logging.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        shutil.copyfile(param_path, log_dir + "_model_params.json")

    if model_class is None:
        model_type = params.pop_choice('model_class', concrete_models.keys())
        model_class = concrete_models[model_type]
    else:
        if params.pop('model_class', None) is not None:
            raise ConfigurationError("You have specified a local model class and passed a model_class argument"
                                     "in the json specification. These options are mutually exclusive.")
    model = model_class(params)

    if model.can_train():
        logger.info("Training model")
        model.train()
        K.clear_session()
    else:
        raise ConfigurationError("The supplied model does not have enough training inputs.")


def load_model(param_path: str, model_class=None):
    """
    Loads and returns a model.

    Parameters
    ----------
    param_path: str, required
        A json file specifying a DeepQaModel.
    model_class: DeepQaModel, optional (default=None)
        This option is useful if you have implemented a new model
        class which is not one of the ones implemented in this library.

    Returns
    -------
    A ``DeepQaModel`` instance.
    """
    param_dict = pyhocon.ConfigFactory.parse_file(param_path)
    params = Params(replace_none(param_dict))
    prepare_environment(params)

    from deep_qa.models import concrete_models
    if model_class is None:
        model_type = params.pop_choice('model_class', concrete_models.keys())
        model_class = concrete_models[model_type]
    else:
        if params.pop('model_class', None) is not None:
            raise ConfigurationError("You have specified a local model class and passed a model_class argument"
                                     "in the json specification. These options are mutually exclusive.")
    model = model_class(params)
    model.load_model()
    return model


def evaluate_model(param_path: str, test: str=False, model_class=None):
    """
    Loads and evaluates a dataset from a saved parameter path.

    Parameters
    ----------
    param_path: str, required
        A json file specifying a DeepQaModel.
    test: str, optional, (default=False)
        Whether to use the test data or validation data to evaluate the model.
    model_class: ``DeepQaModel``, optional (default=None)
        This option is useful if you have implemented a new model class which
        is not one of the ones implemented in this library.

    Returns
    -------
    Numpy arrays of model predictions in the format of model.outputs.

    """
    model = load_model(param_path, model_class=model_class)
    dataset_files = model.test_files if test else model.validation_files
    dataset = model.load_dataset_from_files(dataset_files)
    return model.score_dataset(dataset)
