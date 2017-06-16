from typing import Dict, List, Tuple, Union
import sys
import logging
import os
import json
from copy import deepcopy

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
    params: Params object or dict, required.
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


def run_model_from_file(param_path: str):
    """
    A wrapper around the run_model function which loads json from a file.

    Parameters
    ----------
    param_path: str, required.
        A json paramter file specifying a DeepQA model.
    """
    param_dict = pyhocon.ConfigFactory.parse_file(param_path)
    run_model(param_dict)


def run_model(param_dict: Dict[str, any], model_class=None):
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
    param_dict: Dict[str, any], required.
        A parameter file specifying a DeepQaModel.
    model_class: DeepQaModel, optional (default=None).
        This option is useful if you have implemented a new model class which
        is not one of the ones implemented in this library.
    """
    params = Params(replace_none(param_dict))
    prepare_environment(params)

    # These have to be imported _after_ we set the random seed,
    # because keras uses the numpy random seed.
    from deep_qa.models import concrete_models
    import tensorflow
    from keras import backend as K

    log_dir = params.get("model_serialization_prefix", None)  # pylint: disable=no-member
    if log_dir is not None:
        sys.stdout = TeeLogger(log_dir + "_stdout.log", sys.stdout)
        sys.stderr = TeeLogger(log_dir + "_stderr.log", sys.stderr)
        handler = logging.FileHandler(log_dir + "_python_logging.log")
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
        logging.getLogger().addHandler(handler)
        serialisation_params = deepcopy(params).as_dict(quiet=True)
        with open(log_dir + "_model_params.json", "w") as param_file:
            json.dump(serialisation_params, param_file)

    num_threads = os.environ.get('OMP_NUM_THREADS')
    config = {
            "allow_soft_placement": True,
            "log_device_placement": params.pop("log_device_placement", False)
    }
    if num_threads is not None:
        config["intra_op_parallelism_threads"] = int(num_threads)
    global_session = tensorflow.Session(config=tensorflow.ConfigProto(**config))
    K.set_session(global_session)

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
    logger.info("Loading model from parameter file: %s", param_path)
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


def score_dataset(param_path: str, dataset_files: List[str], model_class=None):
    """
    Loads a model from a saved parameter path and scores a dataset with it, returning the
    predictions.

    Parameters
    ----------
    param_path: str, required
        A json file specifying a DeepQaModel.
    dataset_files: List[str]
        A list of dataset files to score, the same as you would have specified as ``train_files``
        or ``test_files`` in your parameter file.
    model_class: DeepQaModel, optional (default=None)
        This option is useful if you have implemented a new model class which
        is not one of the ones implemented in this library.

    Returns
    -------
    predictions: numpy.array
        Numpy array of model predictions in the format of model.outputs (typically one array, but
        could be List[numpy.array] if your model has multiple outputs).
    labels: numpy.array
        The labels on the dataset, as read by the model.  We return this so you can compute
        whatever metrics you want, if the data was labeled.
    """
    model = load_model(param_path, model_class=model_class)
    dataset = model.load_dataset_from_files(dataset_files)
    return model.score_dataset(dataset)


def evaluate_model(param_path: str, dataset_files: List[str]=None, model_class=None):
    """
    Loads a model and evaluates it on some test set.

    Parameters
    ----------
    param_path: str, required
        A json file specifying a DeepQaModel.
    dataset_files: List[str], optional, (default=None)
        A list of dataset files to evaluate on.  If this is ``None``, we'll evaluate from the
        ``test_files`` parameter in the input files.  If that's also ``None``, we'll crash.
    model_class: DeepQaModel, optional (default=None)
        This option is useful if you have implemented a new model class which
        is not one of the ones implemented in this library.

    Returns
    -------
    Numpy arrays of model predictions in the format of model.outputs.

    """
    model = load_model(param_path, model_class=model_class)
    if dataset_files is None:
        dataset_files = model.test_files
    model.evaluate_model(dataset_files)


def score_dataset_with_ensemble(param_paths: List[str],
                                dataset_files: List[str],
                                model_class=None) -> Tuple[numpy.array, numpy.array]:
    """
    Loads all of the models specified in ``param_paths``, uses each of them to score the dataset
    specified by ``dataset_files``, and averages their scores, return an array of ensembled model
    predictions.

    Parameters
    ----------
    param_paths: List[str]
        A list of parameter files that were used to train models.  You must have already trained
        the corresponding model, as we'll load it and use it in an ensemble here.
    dataset_files: List[str]
        A list of dataset files to score, the same as you would have specified as ``test_files`` in
        any one of the model parameter files.
    model_class: ``DeepQaModel``, optional (default=None)
        This option is useful if you have implemented a new model class which is not one of the
        ones implemented in this library.

    Returns
    -------
    predictions: numpy.array
        Numpy array of model predictions in the format of model.outputs (typically one array, but
        could be List[numpy.array] if your model has multiple outputs).
    labels: numpy.array
        The labels on the dataset, as read by the first model.  We return this so you can compute
        whatever metrics you want, if the data was labeled.  Note that if your models all represent
        that data differently, this will only give the first one.  Hopefully the representation of
        the labels is consistent across the models, though; if not, the whole idea of ensembling
        them this way is moot, anyway.
    """
    models = [load_model(param_path, model_class) for param_path in param_paths]
    predictions = []
    labels_to_return = None
    for i, model in enumerate(models):
        logger.info("Scoring model %d of %d", i + 1, len(models))
        dataset = model.load_dataset_from_files(dataset_files)
        model_predictions, labels = model.score_dataset(dataset)
        predictions.append(model_predictions)
        if labels_to_return is None:
            labels_to_return = labels
    logger.info("Averaging model predictions")
    all_predictions = numpy.stack(predictions)
    averaged = numpy.mean(all_predictions, axis=0)
    return averaged, labels_to_return


def compute_accuracy(predictions: numpy.array, labels: numpy.array):
    """
    Computes a simple categorical accuracy metric, useful if you used ``score_dataset`` to get
    predictions.
    """
    accuracy = numpy.mean(numpy.equal(numpy.argmax(predictions, axis=-1),
                                      numpy.argmax(labels, axis=-1)))
    logger.info("Accuracy: %f", accuracy)
    return accuracy
