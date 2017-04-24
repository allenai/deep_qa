import logging
import math
import os
from typing import Any, Dict, List, Tuple

import numpy
from keras.models import model_from_json
from keras.callbacks import LambdaCallback, TensorBoard, EarlyStopping, CallbackList, ModelCheckpoint

from ..common.checks import ConfigurationError
from ..common.params import Params
from ..data.dataset import Dataset, IndexedDataset
from ..data.instances.instance import Instance
from ..layers.wrappers import OutputMask
from .models import DeepQaModel
from .optimizers import optimizer_from_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Trainer:
    """
    A Trainer object specifies data, a model, and a way to train the model with the data.  Here we
    group all of the common code related to these things, making only minimal assumptions about
    what kind of data you're using or what the structure of your model is.

    The main benefits of this class are having a common place for setting parameters related to
    training, actually running the training with those parameters, and code for saving and loading
    models.

    The intended use of this class is that you construct a subclass that defines a model,
    overriding the abstract methods and (optionally) some of the protected methods in this class.
    Thus there are four kinds of methods in this class: (1) public methods, that are typically only
    used by ``scripts/run_model.py`` (or some other driver that you create), (2) abstract methods
    (beginning with ``_``), which `must` be overridden by any concrete subclass, (3) protected
    methods (beginning with ``_``) that you are meant to override in concrete subclasses, and (4)
    private methods (beginning with ``__``) that you should not need to mess with.  We only include
    the first three in the public docs.

    Parameters
    ----------
    train_files: List[str], optional (default=None)
        The files containing the data that should be used for training.  See
        :func:`~Trainer.load_dataset_from_files()` for more information.
    validation_files: List[str], optional (default=None)
        The files containing the data that should be used for validation, if you do not want to use
        a split of the training data for validation.  The default of None means to just use the
        `validation_split` parameter to split the training data for validation.
    test_files: List[str], optional (default=None)
        The files containing the data that should be used for evaluation.  The default of None
        means to just not perform test set evaluation.
    max_training_instances: int, optional (default=None)
        Upper limit on the number of training instances.  If this is set, and we get more than
        this, we will truncate the data.  Mostly useful for testing things out on small datasets
        before running them on large datasets.
    max_validation_instances: int, optional (default=None)
        Upper limit on the number of validation instances, analogous to ``max_training_instances``.
    max_test_instances: int, optional (default=None)
        Upper limit on the number of test instances, analogous to ``max_training_instances``.
    train_steps_per_epoch: int, optional (default=None)
        If :func:`~Trainer.create_data_arrays` returns a generator instead of actual arrays, how
        many steps should we run from this generator before declaring an "epoch" finished?  The
        default here is reasonable - if this is None, we will set it from the data.
    validation_steps: int, optional (default=None)
        Like ``train_steps_per_epoch``, but for validation data.
    test_steps: int, optional (default=None)
        Like ``train_steps_per_epoch``, but for test data.
    save_models: bool, optional (default=True)
        Should we save the models that we train?  If this is True, you are required to also set the
        model_serialization_prefix parameter, or the code will crash.
    model_serialization_prefix: str, optional (default=None)
        Prefix for saving and loading model files.  Must be set if ``save_models`` is ``True``.
    batch_size: int, optional (default=32)
        Batch size to use when training.
    num_epochs: int, optional (default=20)
        Number of training epochs.
    validation_split: float, optional (default=0.1)
        Amount of training data to use for validation.  If ``validation_files`` is not set, we will
        split the training data into train/dev, using this proportion as dev.  If
        ``validation_files`` is set, this parameter gets ignored.
    optimizer: str or Dict[str, Any], optional (default='adam')
        If this is a str, it must correspond to an optimizer available in Keras (see the list in
        :mod:`deep_qa.training.optimizers`).  If it is a dictionary, it must contain a "type" key,
        with a value that is one of the optimizers in that list.  The remaining parameters in the
        dict are passed as kwargs to the optimizer's constructor.
    loss: str, optional (default='categorical_crossentropy')
        The loss function to pass to ``model.fit()``.  This is currently limited to only loss
        functions that are available as strings in Keras.  If you want to use a custom loss
        function, simply override ``self.loss`` in the constructor of your model, after the call to
        ``super().__init__``.
    metrics: List[str], optional (default=['accuracy'])
        The metrics to evaluate and print after each epoch of training.  This is currently limited
        to only loss functions that are available as strings in Keras.  If you want to use a custom
        metric, simply override ``self.metrics`` in the constructor of your model, after the call
        to ``super().__init__``.
    validation_metric: str, optional (default='val_acc')
        Metric to monitor on the validation data for things like early stopping and saving the best
        model.
    patience: int, optional (default=1)
        Number of epochs to be patient before early stopping.  I.e., if the ``validation_metric``
        does not improve for this many epochs, we will stop training.
    fit_kwargs: Dict[str, Any], optional (default={})
        A dict of additional arguments to Keras' ``model.fit()`` method, in case you want to set
        something that we don't already have options for. These get added to the options already
        captured by other arguments.
    tensorboard_log: str, optional (default=None)
        If set, we will output tensorboard log information here.
    tensorboard_histogram_freq: int, optional (default=0)
        Tensorboard histogram frequency: note that activating the tensorboard histgram (frequency >
        0) can drastically increase model training time.  Please set frequency with consideration
        to desired runtime.
    debug: Dict[str, Any], optional (default={})
        This should be a dict, containing the following keys:

        - "layer_names", which has as a value a list of names that must match layer names in the
          model built by this Trainer.
        - "data", which has as a value either "training", "validation", or a list of file names.
          If you give "training" or "validation", we'll use those datasets, otherwise we'll load
          data from the provided files.  Note that currently "validation" only works if you provide
          validation files, not if you're just using Keras to split the training data.
        - "masks", an optional key that functions identically to "layer_names", except we output
          the mask at each layer given here.

    show_summary_with_masking_info: bool, optional (default=False)
        This is a debugging setting, mostly - we have written a custom model.summary() method that
        supports showing masking info, to help understand what's going on with the masks.
    """
    def __init__(self, params: Params):
        self.name = "Trainer"

        # Data specification parameters.
        self.train_files = params.pop('train_files', None)
        self.validation_files = params.pop('validation_files', None)
        self.test_files = params.pop('test_files', None)
        self.max_training_instances = params.pop('max_training_instances', None)
        self.max_validation_instances = params.pop('max_validation_instances', None)
        self.max_test_instances = params.pop('max_test_instances', None)

        # Data generator parameters.
        self.train_steps_per_epoch = params.pop('train_steps_per_epoch', None)
        self.validation_steps = params.pop('train_steps_per_epoch', None)
        self.test_steps = params.pop('train_steps_per_epoch', None)

        # Model serialization parameters.
        self.save_models = params.pop('save_models', True)
        self.model_prefix = params.pop('model_serialization_prefix', None)
        if self.model_prefix:
            parent_directory = os.path.dirname(self.model_prefix)
            os.makedirs(parent_directory, exist_ok=True)

        # `model.fit()` parameters.
        self.validation_split = params.pop('validation_split', 0.1)
        self.batch_size = params.pop('batch_size', 32)
        self.num_epochs = params.pop('num_epochs', 20)
        self.optimizer = optimizer_from_params(params.pop('optimizer', 'adam'))
        self.loss = params.pop('loss', 'categorical_crossentropy')
        self.metrics = params.pop('metrics', ['accuracy'])
        self.validation_metric = params.pop('validation_metric', 'val_acc')
        self.patience = params.pop('patience', 1)
        self.fit_kwargs = params.pop('fit_kwargs', {})

        # Debugging / logging / misc parameters.
        self.tensorboard_log = params.pop('tensorboard_log', None)
        self.tensorboard_histogram_freq = params.pop('tensorboard_histogram_freq', 0)
        self.debug_params = params.pop('debug', {})
        self.show_summary_with_masking = params.pop('show_summary_with_masking_info', False)

        # We've now processed all of the parameters, and we're the base class, so there should not
        # be anything left.
        params.assert_empty("Trainer")

        # Model-specific member variables that will get set and used later.
        self.model = None
        self.debug_model = None

        # Should we update state when loading the training data in `self.train()`?  Generally, yes,
        # you need to do this.  But if you've loaded a pre-trained model, the model state has
        # already been frozen, and trying to update things like the vocabulary will break things.
        # So we set this to false when loading a saved model.
        self.update_model_state_with_training_data = True

        # Training-specific member variables that will get set and used later.
        self.best_epoch = -1

        # We store the datasets used for training and validation, both before processing and after
        # processing, in case a subclass wants to modify it between epochs for whatever reason.
        self.training_dataset = None
        self.training_arrays = None

        self.validation_dataset = None
        self.validation_arrays = None

        self.test_dataset = None
        self.test_arrays = None

        self.debug_dataset = None
        self.debug_arrays = None

    ################
    # Public methods
    ################

    def can_train(self):
        return self.train_files is not None

    def load_data_arrays(self,
                         data_files: List[str],
                         max_instances: int=None) -> Tuple[Dataset, numpy.array, numpy.array]:
        """
        Loads a :class:`Dataset` from a list of files, then converts it into numpy arrays for
        both inputs and outputs, returning all three of these to you.  This literally just calls
        ``self.load_dataset_from_files``, then ``self.create_data_arrays``; it's just a convenience
        method if you want to do both of these at the same time, and also lets you truncate the
        dataset if you want.

        Note that if you have any kind of state in your model that depends on a training dataset
        (e.g., a vocabulary, or padding dimensions) those must be set prior to calling this method.

        Parameters
        ----------
        data_files: List[str]
            The files to load.  These will get passed to ``self.load_dataset_from_files()``, which
            subclasses must implement.
        max_instances: int, optional (default=None)
            If not ``None``, we will restrict the dataset to only this many instances.  This is
            mostly useful for testing models out on subsets of your data.

        Returns
        -------
        dataset: Dataset
            A :class:`Dataset` object containing the instances read from the data files
        input_arrays: numpy.array
            An array or tuple of arrays suitable to be passed as inputs ``x`` to Keras'
            ``model.fit(x, y)``, ``model.evaluate(x, y)`` or ``model.predict(x)`` methods
        label_arrays: numpy.array
            An array or tuple of arrays suitable to be passed as outputs ``y`` to Keras'
            ``model.fit(x, y)`` or ``model.evaluate(x, y)`` methods
        """
        logger.info("Loading data from %s", str(data_files))
        dataset = self.load_dataset_from_files(data_files)
        if max_instances is not None:
            logger.info("Truncating the dataset to %d instances", max_instances)
            dataset = dataset.truncate(max_instances)
        logger.info("Indexing dataset")
        indexing_kwargs = self._dataset_indexing_kwargs()
        indexed_dataset = dataset.to_indexed_dataset(**indexing_kwargs)
        data_arrays = self.create_data_arrays(indexed_dataset)
        return (dataset, data_arrays)

    def train(self):
        '''
        Trains the model.

        All training parameters have already been passed to the constructor, so we need no
        arguments to this method.
        '''
        logger.info("Running training (%s)", self.name)

        # First we need to prepare the data that we'll use for training.  For the training data, we
        # might need to update model state based on this dataset, so we handle it differently than
        # we do the validation and training data.
        self.training_dataset = self.load_dataset_from_files(self.train_files)
        if self.max_training_instances:
            self.training_dataset = self.training_dataset.truncate(self.max_training_instances)
        if self.update_model_state_with_training_data:
            self.set_model_state_from_dataset(self.training_dataset)
        logger.info("Indexing training data")
        indexing_kwargs = self._dataset_indexing_kwargs()
        indexed_training_dataset = self.training_dataset.to_indexed_dataset(**indexing_kwargs)
        if self.update_model_state_with_training_data:
            self.set_model_state_from_indexed_dataset(indexed_training_dataset)
        self.training_arrays = self.create_data_arrays(indexed_training_dataset)

        if self.validation_files:
            self.validation_dataset, self.validation_arrays = self.load_data_arrays(self.validation_files)
        if self.test_files:
            self.test_dataset, self.test_arrays = self.load_data_arrays(self.test_files)

        # Then we build the model and compile it.
        logger.info("Building the model")
        self.model = self._build_model()
        self.model.summary(show_masks=self.show_summary_with_masking)
        self.model.compile(**self.__compile_kwargs())

        if self.debug_params:
            # Get the list of layers whose outputs will be visualized as per the
            # solver definition and build a debug model.
            debug_layer_names = self.debug_params['layer_names']
            debug_masks = self.debug_params.get('masks', [])
            debug_data = self.debug_params['data']
            if debug_data == "training":
                self.debug_dataset = self.training_dataset
                self.debug_arrays = self.training_arrays
            elif debug_data == "validation":
                # NOTE: This currently only works if you've specified specific validation data, not
                # if you are just splitting the training data for validation.
                self.debug_dataset = self.validation_dataset
                self.debug_arrays = self.validation_arrays
            else:
                # If the `data` param is not "training" or "validation", we assume it's a list of
                # file names.
                self.debug_dataset, self.debug_arrays = self.load_data_arrays(debug_data)
            self.debug_model = self.__build_debug_model(debug_layer_names, debug_masks)

        # Now we actually train the model using various Keras callbacks to control training.
        callbacks = self._get_callbacks()
        kwargs = {'epochs': self.num_epochs, 'callbacks': [callbacks], 'batch_size': self.batch_size}
        # We'll check for explicit validation data first; if you provided this, you definitely
        # wanted to use it for validation.  self.validation_split is non-zero by default,
        # so you may have left it above zero on accident.
        if self.validation_arrays is not None:
            kwargs['validation_data'] = self.validation_arrays
        elif self.validation_split > 0.0:
            kwargs['validation_split'] = self.validation_split

        # Add the user-specified arguments to fit.
        kwargs.update(self.fit_kwargs)
        # We now pass all the arguments to the model's fit function, which does all of the training.
        if isinstance(self.training_arrays, tuple):
            history = self.model.fit(self.training_arrays[0], self.training_arrays[1], **kwargs)
        else:
            # If the data was produced by a generator, we have a bit more work to do to get the
            # arguments right.
            kwargs.pop('batch_size')
            kwargs['steps_per_epoch'] = self.train_steps_per_epoch
            if kwargs['steps_per_epoch'] is None:
                kwargs['steps_per_epoch'] = math.ceil(len(self.training_dataset.instances) / self.batch_size)
            if self.validation_arrays is not None and not isinstance(self.validation_arrays, tuple):
                kwargs['validation_steps'] = self.validation_steps
                if kwargs['validation_steps'] is None:
                    kwargs['validation_steps'] = math.ceil(len(self.validation_dataset.instances) /
                                                           self.batch_size)
            history = self.model.fit_generator(self.training_arrays, **kwargs)

        # After finishing training, we save the best weights and
        # any auxillary files, such as the model config.
        self.best_epoch = int(numpy.argmax(history.history[self.validation_metric]))
        if self.save_models:
            self.__save_best_model()
            self._save_auxiliary_files()

        # If there are test files, we evaluate on the test data.
        if self.test_files:
            self.load_model()
            logger.info("Evaluting model on the test set.")
            if isinstance(self.test_arrays, tuple):
                scores = self.model.evaluate(self.test_arrays[0], self.test_arrays[1])
            else:
                test_steps = self.test_steps
                if test_steps is None:
                    test_steps = math.ceil(len(self.test_dataset.instances) / self.batch_size)
                scores = self.model.evaluate_generator(self.test_arrays, test_steps)
            for idx, metric in enumerate(self.model.metrics_names):
                print("{}: {}".format(metric, scores[idx]))

    def score_dataset(self, dataset: Dataset):
        inputs, _ = self.create_data_arrays(dataset)
        return self.model.predict(inputs)

    def load_model(self, epoch: int=None):
        """
        Loads a serialized model, using the ``model_serialization_prefix`` that was passed to the
        constructor.  If epoch is not None, we try to load the model from that epoch.  If epoch is
        not given, we load the best saved model.
        """
        logger.info("Loading serialized model")
        # Loading serialized model
        model_config_file = open("%s_config.json" % self.model_prefix)
        model_config_json = model_config_file.read()
        model_config_file.close()
        self.model = model_from_json(model_config_json,
                                     custom_objects=self._get_custom_objects())
        if epoch is not None:
            model_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, epoch)
        else:
            model_file = "%s_weights.h5" % self.model_prefix
        logger.info("Loading weights from file %s", model_file)
        self.model.load_weights(model_file)
        self.model.summary(show_masks=self.show_summary_with_masking)
        self._load_auxiliary_files()
        self._set_params_from_model()
        self.model.compile(**self.__compile_kwargs())
        self.update_model_state_with_training_data = False

    ##################
    # Abstract methods - you MUST override these
    ##################

    def load_dataset_from_files(self, files: List[str]) -> Dataset:
        """
        Given a list of file inputs, load a raw dataset from the files.  This is a list because
        some datasets are specified in more than one file (e.g., a file containing the instances,
        and a file containing background information about those instances).
        """
        raise NotImplementedError

    def set_model_state_from_dataset(self, dataset: Dataset):
        """
        Given a raw :class:`Dataset` object, set whatever model state is necessary.  The most
        obvious use case for this is for computing a vocabulary in
        :class:`~deep_qa.training.text_trainer.TextTrainer`.  Note that this is not an
        :class:`IndexedDataset`, and you should not make it one.  Use
        :func:`~Trainer.set_model_state_from_indexed_dataset()` for setting state that depends on
        the data having already been indexed; otherwise you'll duplicate the work of doing the
        indexing.
        """
        raise NotImplementedError

    def set_model_state_from_indexed_dataset(self, dataset: IndexedDataset):
        """
        Given an :class:`IndexedDataset`, set whatever model state is necessary.  This is typically
        stuff around padding.
        """
        raise NotImplementedError

    def create_data_arrays(self, dataset: IndexedDataset) -> Tuple[numpy.array, numpy.array]:
        """
        Takes a raw dataset and converts it into training inputs and labels that can be used to
        either train a model or make predictions.  Depending on parameters passed to the
        constructor of this ``Trainer``, this could either return two actual array objects, or a
        single generator that generates batches of two array objects.

        Parameters
        ----------
        dataset: Dataset
            A ``Dataset`` of the same format as read by ``load_dataset_from_files()`` (we will
            call this directly with the output from that method, in fact)

        Returns
        -------
        input_arrays: numpy.array or Tuple[numpy.array]
        label_arrays: numpy.array, Tuple[numpy.array], or None
        generator: a Python generator returning Tuple[input_arrays, label_arrays]
            If this is returned, it is the only return value.  We `either` return a
            ``Tuple[input_arrays, label_arrays]``, `or` this generator.
        """
        raise NotImplementedError

    def _build_model(self) -> DeepQaModel:
        """Constructs and returns a DeepQaModel (which is a wrapper around a Keras Model) that will
        take the output of self._get_training_data as input, and produce as output a true/false
        decision for each input.

        The returned model will be used to call model.fit(train_input, train_labels).
        """
        raise NotImplementedError

    def _set_params_from_model(self):
        """
        Called after a model is loaded, this lets you update member variables that contain model
        parameters, like max sentence length, that are not stored as weights in the model object.
        This is necessary if you want to process a new data instance to be compatible with the
        model for prediction, for instance.
        """
        raise NotImplementedError

    def _dataset_indexing_kwargs(self) -> Dict[str, Any]:
        """
        In order to index a dataset, we may need some parameters (e.g., an object that stores the
        vocabulary of your model, in order to convert words into indices).  You can pass those
        here, or return an emtpy dictionary if there's nothing.  These will get passed to
        :func:`Dataset.to_indexed_dataset`.
        """
        raise NotImplementedError

    ###################
    # Protected methods - you CAN override these, if you want
    ###################

    def _get_callbacks(self):
        """
         Returns a set of Callbacks which are used to perform various functions within Keras' .fit method.
         Here, we use an early stopping callback to add patience with respect to the validation metric and
         a Lambda callback which performs the model specific callbacks which you might want to build into
         a model, such as re-encoding some background knowledge.

         Additionally, there is also functionality to create Tensorboard log files. These can be visualised
         using 'tensorboard --logdir /path/to/log/files' after training.
        """
        early_stop = EarlyStopping(monitor=self.validation_metric, patience=self.patience)
        model_callbacks = LambdaCallback(on_epoch_begin=lambda epoch, logs: self._pre_epoch_hook(epoch),
                                         on_epoch_end=lambda epoch, logs: self._post_epoch_hook(epoch))
        callbacks = [early_stop, model_callbacks]

        if self.tensorboard_log is not None:
            tensorboard_visualisation = TensorBoard(log_dir=self.tensorboard_log,
                                                    histogram_freq=self.tensorboard_histogram_freq)
            callbacks.append(tensorboard_visualisation)

        if self.debug_params:
            debug_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                            self.__debug(self.debug_params["layer_names"],
                                                         self.debug_params.get("masks", []), epoch))
            callbacks.append(debug_callback)
            return CallbackList(callbacks)

        # Some witchcraft is happening here - we don't specify the epoch replacement variable
        # checkpointing string, because Keras does that within the callback if we specify it here.
        if self.save_models:
            checkpointing = ModelCheckpoint(self.model_prefix + "_weights_epoch={epoch:d}.h5",
                                            save_best_only=True, save_weights_only=True,
                                            monitor=self.validation_metric)
            callbacks.append(checkpointing)

        return CallbackList(callbacks)

    def _pre_epoch_hook(self, epoch: int):
        """
        This method gets called before each epoch of training.  If you want to do any kind of
        processing in between epochs (e.g., updating the training data for whatever reason), here
        is your chance to do so.
        """
        pass

    def _post_epoch_hook(self, epoch: int):
        """
        This method gets called directly after model.fit(), before making any early stopping
        decisions.  If you want to modify anything after each iteration (e.g., computing a
        different kind of validation loss to use for early stopping, or just computing and printing
        accuracy on some other held out data), you can do that here. If you require extra parameters,
        use calls to local methods rather than passing new parameters, as this hook is run via a
        Keras Callback, which is fairly strict in it's interface.
        """
        pass

    def _output_debug_info(self, output_dict: Dict[str, numpy.array], epoch: int):
        logger.info("Outputting debug results")
        debug_output_file = open("%s_debug_%d.txt" % (self.model_prefix, epoch), "w")
        overall_debug_info = self._overall_debug_output(output_dict)
        debug_output_file.write(overall_debug_info)
        for instance_index, instance in enumerate(self.debug_dataset.instances):
            instance_output_dict = {}
            for layer_name, output in output_dict.items():
                if layer_name == 'masks':
                    instance_output_dict['masks'] = {}
                    for mask_name, mask_output in output.items():
                        instance_output_dict['masks'][mask_name] = mask_output[instance_index]
                else:
                    instance_output_dict[layer_name] = output[instance_index]
            instance_info = self._instance_debug_output(instance, instance_output_dict)
            debug_output_file.write(instance_info + '\n')
        debug_output_file.close()

    def _overall_debug_output(self, output_dict: Dict[str, numpy.array]) -> str: # pylint: disable=unused-argument
        return "Number of instances: %d\n" % len(self.debug_dataset.instances)

    def _instance_debug_output(self, instance: Instance, outputs: Dict[str, numpy.array]) -> str:
        """
        This method takes an Instance and all of the debug outputs for that Instance, puts them
        into some human-readable format, and returns that as a string.  `outputs` will have one key
        corresponding to each item in the `debug.layer_names` parameter given to the constructor of
        this object.

        The default here is `pass` instead of `raise NotImplementedError`, because you're not
        required to implement debugging for your model.
        """
        pass

    def _load_auxiliary_files(self):
        """
        Called during model loading.  If you have some auxiliary pickled object, such as an object
        storing the vocabulary of your model, you can load it here.
        """
        pass

    def _save_auxiliary_files(self):
        """
        Called after training. If you have some auxiliary object, such as an object storing
        the vocabulary of your model, you can save it here. The model config is saved by default.
        """
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json" % (self.model_prefix), "w")
        print(model_config, file=model_config_file)
        model_config_file.close()

    @classmethod
    def _get_custom_objects(cls):
        """
        If you've used any Layers that Keras doesn't know about, you need to specify them in this
        dictionary, so we can load them correctly.
        """
        return {
                "DeepQaModel": DeepQaModel
        }

    #################
    # Private methods - you can't to override these.  If you find yourself needing to, we can
    # consider making them protected instead.
    #################

    def __save_best_model(self):
        """
        Copies the weights from the best epoch to a final weight file.

        The point of this is so that the input/output spec of the NNSolver is simpler.  Someone
        calling this as a subroutine doesn't have to worry about which epoch ended up being the
        best, they can just use the final weight file.  You can still use models from other epochs
        if you really want to.
        """
        from shutil import copyfile
        epoch_weight_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, self.best_epoch)
        final_weight_file = "%s_weights.h5" % self.model_prefix
        copyfile(epoch_weight_file, final_weight_file)
        logger.info("Saved the best model to %s", final_weight_file)

    def __build_debug_model(self, debug_layer_names: List[str], debug_masks: List[str]):
        """
        Here we build a very simple kind of debug model: one that takes the same inputs as
        self.model, and runs the model up to some particular layers, and outputs the values at
        those layers.

        In addition, you can optionally specify some number of layers for which you want to output
        the mask computed by that layer.

        If you want something more complicated, override this method.
        """
        debug_inputs = self.model.get_input_at(0)  # list of all input_layers
        debug_output_dict = {}
        layer_names = set(debug_layer_names)
        mask_names = set(debug_masks)
        for layer in self.model.layers:
            if layer.name in layer_names:
                debug_output_dict[layer.name] = layer.get_output_at(0)
                layer_names.remove(layer.name)
            if layer.name in mask_names:
                mask = OutputMask()(layer.get_output_at(0))
                debug_output_dict['mask_for_' + layer.name] = mask
                mask_names.remove(layer.name)
        if len(layer_names) != 0 or len(mask_names):
            raise ConfigurationError("Unmatched debug layer names: " + str(layer_names | mask_names))
        # The outputs need to be in the same order as `debug_layer_names`, or downstream code will
        # have issues.
        debug_outputs = [debug_output_dict[name] for name in debug_layer_names]
        debug_outputs.extend([debug_output_dict['mask_for_' + name] for name in debug_masks])
        return DeepQaModel(input=debug_inputs, output=debug_outputs)

    def __debug(self, debug_layer_names: List[str], debug_masks: List[str], epoch: int):
        """
        Runs the debug model and saves the results to a file.
        """
        logger.info("Running debug model")
        # Shows intermediate outputs of the model on validation data
        outputs = self.debug_model.predict(self.debug_arrays[0])
        output_dict = {}
        if len(debug_layer_names) == 1:
            output_dict[debug_layer_names[0]] = outputs
        else:
            for layer_name, output in zip(debug_layer_names, outputs[:len(debug_layer_names)]):
                output_dict[layer_name] = output
        for layer_name, output in zip(debug_masks, outputs[len(debug_layer_names):]):
            if 'masks' not in output_dict:
                output_dict['masks'] = {}
            output_dict['masks'][layer_name] = output
        self._output_debug_info(output_dict, epoch)

    def __compile_kwargs(self):
        """
        Because we call model.compile() in a few different places in the code, and we have a few
        member variables that we use to set arguments for model.compile(), we group those arguments
        together here, to only specify them once.
        """
        return {
                'loss': self.loss,
                'optimizer': self.optimizer,
                'metrics': self.metrics,
                }
