import logging
import os
from typing import Any, Dict, List

import numpy
import keras.backend as K
from keras.models import model_from_json
from keras.callbacks import LambdaCallback, TensorBoard, EarlyStopping, CallbackList, ModelCheckpoint

from . import concrete_pretrainers
from ..common.checks import ConfigurationError
from ..common.params import get_choice
from ..data.dataset import Dataset
from ..data.instances.instance import Instance
from ..layers.wrappers.output_mask import OutputMask
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
    """
    def __init__(self, params: Dict[str, Any]):
        self.name = "Trainer"

        # Should we save the models that we train?  If this is True, you are required to also set
        # the model_serialization_prefix parameter, or the code will crash.
        self.save_models = params.pop('save_models', True)

        # Prefix for saving and loading model files
        self.model_prefix = params.pop('model_serialization_prefix', None)
        if self.model_prefix:
            parent_directory = os.path.dirname(self.model_prefix)
            os.makedirs(parent_directory, exist_ok=True)

        # Preferred backend to use for training. If a different backend is detected, we still
        # train but we also warn the user.
        self.preferred_backend = params.pop('preferred_backend', None)
        if self.preferred_backend and self.preferred_backend.lower() != K.backend():
            warning_message = self._make_backend_warning(self.preferred_backend.lower(),
                                                         K.backend())
            logger.warning(warning_message)

        self.batch_size = params.pop('batch_size', 32)
        # Upper limit on the number of training instances.  If this is set, and we get more than
        # this, we will truncate the data.
        self.max_training_instances = params.pop('max_training_instances', None)
        # Amount of training data to use for Keras' validation (not our QA validation, set by
        # the validation_file param, which is separate).  This value is passed as
        # 'validation_split' to Keras' model.fit().
        self.keras_validation_split = params.pop('keras_validation_split', 0.1)
        # Number of train epochs.
        self.num_epochs = params.pop('num_epochs', 20)
        # Number of epochs to be patient before early stopping.
        self.patience = params.pop('patience', 1)
        # Log directory for tensorboard.
        self.tensorboard_log = params.pop('tensorboard_log', None)
        # Tensorboard histogram frequency: note that activating the tensorboard histgram (frequency > 0) can
        # drastically increase model training time.  Please set frequency with consideration to desired runtime.
        self.tensorboard_histogram_freq = params.pop('tensorboard_histogram_freq', 0)

        # The files containing the data that should be used for training.  See
        # _load_dataset_from_files().
        self.train_files = params.pop('train_files', None)

        # The files containing the data that should be used for validation, if you do not want to
        # use a split of the training data for validation.  The default of None means to just use
        # the `keras_validation_split` parameter to split the training data for validation.
        self.validation_files = params.pop('validation_files', None)

        optimizer_params = params.pop('optimizer', 'adam')
        self.optimizer = optimizer_from_params(optimizer_params)
        self.loss = params.pop('loss', 'categorical_crossentropy')
        self.metrics = params.pop('metrics', ['accuracy'])
        self.validation_metric = params.pop('validation_metric', 'val_acc')

        # A dict of additional arguments to the fit method. These get added to
        # the options already captured by other arguments.
        self.fit_kwargs = params.pop('fit_kwargs', {})

        # This is a debugging setting, mostly - we have written a custom model.summary() method
        # that supports showing masking info, to help understand what's going on with the masks.
        self.show_summary_with_masking = params.pop('show_summary_with_masking_info', False)

        # This should be a dict, containing the following keys:
        # - "layer_names", which has as a value a list of names that must match layer names in the
        #     model build by this Trainer.
        # - "data", which has as a value either "training", "validation", or a list of file names.
        #     If you give "training" or "validation", we'll use those datasets, otherwise we'll
        #     load data from the provided files.  Note that currently "validation" only works if
        #     you provide validation files, not if you're just using Keras to split the training
        #     data.
        # - "masks", an optional key that functions identically to "layer_names", except we output
        #     the mask at each layer given here.
        self.debug_params = params.pop('debug', {})

        pretrainer_params = params.pop('pretrainers', [])
        self.pretrainers = []
        for pretrainer_param in pretrainer_params:
            pretrainer_type = get_choice(pretrainer_param, "type", concrete_pretrainers.keys())
            pretrainer = concrete_pretrainers[pretrainer_type](self, pretrainer_param)
            self.pretrainers.append(pretrainer)

        # We've now processed all of the parameters, and we're the base class, so there should not
        # be anything left.
        if len(params.keys()) != 0:
            raise ConfigurationError("You passed unrecognized parameters: " + str(params))

        # Model-specific member variables that will get set and used later.
        self.model = None
        self.debug_model = None

        # Training-specific member variables that will get set and used later.
        self.best_epoch = -1

        # We store the datasets used for training and validation, both before processing and after
        # processing, in case a subclass wants to modify it between epochs for whatever reason.
        self.training_dataset = None
        self.train_input = None
        self.train_labels = None
        self.validation_dataset = None
        self.validation_input = None
        self.validation_labels = None
        self.debug_dataset = None
        self.debug_input = None

    def can_train(self):
        return self.train_files is not None

    def _load_dataset_from_files(self, files: List[str]) -> Dataset:
        """
        Given a list of file inputs, load a raw dataset from the files.  This is a list because
        some datasets are specified in more than one file (e.g., a file containing the instances,
        and a file containing background information about those instances).
        """
        raise NotImplementedError

    def _prepare_data(self, dataset: Dataset, for_train: bool, update_data_indexer=True):
        """
        Takes a raw dataset and converts it into training inputs and labels that can be used to
        either train a model or make predictions.

        Input: a Dataset of the same format as read by the read_dataset_from_files() method, and an
        indicator for whether this is being done for the training data (so that, e.g., you can set
        model parameters based on characteristics of the training data).

        Output: a tuple of (inputs, labels), which can be fed directly to Keras' model.fit()
        and model.predict() methods.  `labels` is allowed to be None in the second case.
        """
        raise NotImplementedError

    def _prepare_instance(self, instance: Instance, make_batch: bool=True):
        """
        Like self._prepare_data(), but for a single Instance.  Used most often for making
        predictions one at a time on test data (though you should avoid that if possible, as larger
        batches would be more efficient).

        The make_batch argument determines whether we make the return value into a batch or not by
        calling numpy.expand_dims.  Keras' model.predict() method requires a batch, so we need an
        extra dimension.  If you're going to do the batch conversion yourself, or don't need it,
        you can pass False for that parameter.
        """
        raise NotImplementedError

    def _pretrain(self):
        """
        Runs whatever pre-training has been specified in the constructor.
        """
        logger.info("Running pre-training")
        for pretrainer in self.pretrainers:
            pretrainer.train()
        self._pretraining_finished_hook()

    def _process_pretraining_data(self):
        """
        Processes the pre-training data in whatever way you want, typically for setting model
        parameters like vocabulary.  This happens _before_ the training data itself is processed.
        We don't know what processing this might entail, or whether you are even doing any
        pre-training, so we just pass here by default.
        """
        pass

    def _pretraining_finished_hook(self):
        """
        This is called when pre-training finishes (if there were any pre-trainers specified).  You
        can do whatever you want in here, like changing model parameters based on what happened
        during pre-training, or saving a pre-trained model, or whatever.

        Default implementation is to call pretrainer._on_finished() for each pre-trainer, which by
        default is a `pass`.
        """
        for pretrainer in self.pretrainers:
            pretrainer.on_finished()

    def _compile_kwargs(self):
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

    def _build_model(self) -> DeepQaModel:
        """Constructs and returns a DeepQaModel (which is a wrapper around a Keras Model) that will
        take the output of self._get_training_data as input, and produce as output a true/false
        decision for each input.

        The returned model will be used to call model.fit(train_input, train_labels).
        """
        raise NotImplementedError

    def prepare_data(self, train_files, max_training_instances=None,
                     validation_files=None, update_data_indexer=True):
        logger.info("Getting training data")
        training_dataset = self._load_dataset_from_files(train_files)
        if max_training_instances is not None:
            logger.info("Truncating the training dataset to %d instances", max_training_instances)
            training_dataset = training_dataset.truncate(max_training_instances)
        train_input, train_labels = self._prepare_data(training_dataset,
                                                       for_train=True,
                                                       update_data_indexer=update_data_indexer)
        validation_dataset = validation_input = validation_labels = None
        if validation_files:
            logger.info("Getting validation data")
            validation_dataset = self._load_dataset_from_files(validation_files)
            validation_input, validation_labels = self._prepare_data(validation_dataset,
                                                                     for_train=False,
                                                                     update_data_indexer=update_data_indexer)
        return ((training_dataset, train_input, train_labels),
                (validation_dataset, validation_input, validation_labels))

    def train(self):
        '''
        Trains the model.

        All training parameters have already been passed to the constructor, so we need no
        arguments to this method.
        '''
        logger.info("Running training (%s)", self.name)

        # Before actually doing any training, we'll run whatever pre-training has been specified.
        # Note that this can have funny interactions with model parameters that get fit to the
        # training data.  We don't really know here what you want to do with the data you have for
        # pre-training, if any, so we provide a hook that you can override to do whatever you want.
        if self.pretrainers:
            self._process_pretraining_data()

        # First we need to prepare the data that we'll use for training.
        train_data, val_data = self.prepare_data(self.train_files, self.max_training_instances,
                                                 self.validation_files)
        self.training_dataset, self.train_input, self.train_labels = train_data
        self.validation_dataset, self.validation_input, self.validation_labels = val_data

        # We need to actually do pretraining _after_ we've loaded the training data, though, as we
        # need to build the models to be consistent between training and pretraining.  The training
        # data tells us a max sentence length, which we need for the pretrainer.
        if self.pretrainers:
            self._pretrain()

        # Then we build the model and compile it.
        logger.info("Building the model")
        self.model = self._build_model()
        self.model.summary(show_masks=self.show_summary_with_masking)
        self.model.compile(**self._compile_kwargs())

        if self.debug_params:
            # Get the list of layers whose outputs will be visualized as per the
            # solver definition and build a debug model.
            debug_layer_names = self.debug_params['layer_names']
            debug_masks = self.debug_params.get('masks', [])
            debug_data = self.debug_params['data']
            if debug_data == "training":
                self.debug_dataset = self.training_dataset
                self.debug_input = self.train_input
            elif debug_data == "validation":
                # NOTE: This currently only works if you've specified specific validation data, not
                # if you are just splitting the training data for validation.
                self.debug_dataset = self.validation_dataset
                self.debug_input = self.validation_input
            else:
                # If the `data` param is not "training" or "validation", we assume it's a list of
                # file names.
                self.debug_dataset = self._load_dataset_from_files(debug_data)
                self.debug_input, _ = self._prepare_data(self.debug_dataset, for_train=False)
            self.debug_model = self._build_debug_model(debug_layer_names, debug_masks)

        # Now we actually train the model using various Keras callbacks to control training.
        callbacks = self._get_callbacks()
        kwargs = {'nb_epoch': self.num_epochs, 'callbacks': [callbacks], 'batch_size': self.batch_size}
        # We'll check for explicit validation data first; if you provided this, you definitely
        # wanted to use it for validation.  self.keras_validation_split is non-zero by default,
        # so you may have left it above zero on accident.
        if self.validation_input is not None:
            kwargs['validation_data'] = (self.validation_input, self.validation_labels)
        elif self.keras_validation_split > 0.0:
            kwargs['validation_split'] = self.keras_validation_split

        # add the user-specified arguments to fit
        kwargs.update(self.fit_kwargs)
        # We now pass all the arguments to the model's fit function, which does all of the training.
        history = self.model.fit(self.train_input, self.train_labels, **kwargs)
        # After finishing training, we save the best weights and
        # any auxillary files, such as the model config.

        self.best_epoch = int(numpy.argmax(history.history[self.validation_metric]))
        if self.save_models:
            self._save_best_model()
            self._save_auxiliary_files()

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
            if K.backend() == 'theano':
                raise ConfigurationError("Tensorboard logging is only compatibile with Tensorflow. "
                                         "Change the backend using the KERAS_BACKEND environment variable.")
            tensorboard_visualisation = TensorBoard(log_dir=self.tensorboard_log,
                                                    histogram_freq=self.tensorboard_histogram_freq)
            callbacks.append(tensorboard_visualisation)

        if self.debug_params:
            debug_callback = LambdaCallback(on_epoch_end=lambda epoch, logs:
                                            self._debug(self.debug_params["layer_names"],
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

    def _debug(self, debug_layer_names: List[str], debug_masks: List[str], epoch: int):
        """
        Runs the debug model and saves the results to a file.
        """
        logger.info("Running debug model")
        # Shows intermediate outputs of the model on validation data
        outputs = self.debug_model.predict(self.debug_input)
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

    def _build_debug_model(self, debug_layer_names: List[str], debug_masks: List[str]):
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

    def score_dataset(self, dataset: Dataset):
        inputs, _ = self._prepare_data(dataset, False)
        return self.model.predict(inputs)

    def score_instance(self, instance: Instance):
        inputs, _ = self._prepare_instance(instance)
        try:
            return self.model.predict(inputs)
        except:
            print('Inputs were: ' + str(inputs))
            raise

    def load_model(self, epoch: int=None):
        """
        Loads a serialized model.  If epoch is not None, we try to load the model from that epoch.
        If epoch is not given, we load the best saved model.

        Paths in here must match those in self._save_model(epoch) and self._save_best_model(), or
        things will break.
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
        self._load_layers()
        self._load_auxiliary_files()
        self._set_params_from_model()
        self.model.compile(**self._compile_kwargs())

    def _load_layers(self):
        """
        If you want to use member variables that contain Layers after the model is loaded, you need
        to set them from the model.  For instance, say you have an embedding layer for word
        sequences, and you want to take a loaded model, build a sub-model out of it that contains
        the embedding layer, and use that model somehow.  In that case, the member variable for the
        embedding layer needs to be set from the loaded model.  You can do that here.
        """
        pass

    def _load_auxiliary_files(self):
        """
        Called during model loading.  If you have some auxiliary pickled object, such as an object
        storing the vocabulary of your model, you can load it here.
        """
        pass

    def _set_params_from_model(self):
        """
        Called after a model is loaded, this lets you update member variables that contain model
        parameters, like max sentence length, that are not stored as weights in the model object.
        This is necessary if you want to process a new data instance to be compatible with the
        model for prediction, for instance.
        """
        pass

    def _save_best_model(self):
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

    def _save_auxiliary_files(self):
        """
        Called after training. If you have some auxiliary object, such as an object storing
        the vocabulary of your model, you can save it here. The model config is saved by default.
        """
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json" % (self.model_prefix), "w")
        print(model_config, file=model_config_file)
        model_config_file.close()

    @staticmethod
    def _make_backend_warning(preferred_backend, actual_backend):
        warning_info = ("@ Preferred backend is %s, but "
                        "current backend is %s. @" % (preferred_backend,
                                                      actual_backend))
        end_row = "@" * len(warning_info)
        warning_row_spaces = len(warning_info) - len("@ WARNING: @")
        left_warning_row_spaces = right_warning_row_spaces = warning_row_spaces // 2
        if warning_row_spaces % 2 == 1:
            # left and right have uneven spacing
            right_warning_row_spaces += 1
        left_warning_row = "\n@" + " " * left_warning_row_spaces
        right_warning_row = " " * right_warning_row_spaces + "@\n"
        warning_message = ("\n" + end_row +
                           left_warning_row + " WARNING: " + right_warning_row +
                           warning_info +
                           "\n" + end_row)
        return warning_message

    @classmethod
    def _get_custom_objects(cls):
        """
        If you've used any Layers that Keras doesn't know about, you need to specify them in this
        dictionary, so we can load them correctly.
        """
        return {
                "DeepQaModel": DeepQaModel
        }
