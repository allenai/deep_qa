import logging

from keras.callbacks import ModelCheckpoint
from overrides import overrides

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class ReplicaModelCheckpoint(ModelCheckpoint):
    """
    Save the model after every epoch. See the Keras ModelCheckpoint class for more information
    about basic usage.

    This Callback is designed to be used with a ``DeepQaModel`` which was parallelised across
    multiple GPUs. In order to achieve the data parallelism implemented in
    ``:func:~.multi_gpu.make_parallel``, we use lambda layers to split a large batch into
    inputs for the models tiled across the GPUs and to aggregate the model outputs.
    When we are saving the model, we don't want these layers to be present, so we need to
    retrieve the section of the model which we want to serialise.

    """

    def __init__(self, *args, **kwargs):
        super(ReplicaModelCheckpoint, self).__init__(*args, **kwargs)

    @overrides
    def on_epoch_end(self, epoch, logs=None):
        """
        Saves the layer in the multi-gpu model corresponding to the actual DeepQaModel.
        This layer is always at the -(num_lambda_layers+1)-th index, (the layer after the number
        of Lambda layers we used to split the data up into batches).
        """
        logs = logs or {}

        # We have a Lambda layer per GPU, so the total number
        # is simply the number of model outputs.
        num_lambda_layers = len(self.model.outputs)

        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    logger.warning('Can save best model only with %s available, '
                                   'skipping.', self.monitor, exec_info=RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            logger.debug('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                         ' saving model to %s', epoch, self.monitor, self.best,
                                         current, filepath)
                        self.best = current
                        if self.save_weights_only:
                            self.model.layers[-(num_lambda_layers+1)].save_weights(filepath, overwrite=True)
                        else:
                            self.model.layers[-(num_lambda_layers+1)].save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            logger.debug('Epoch %05d: %s did not improve', epoch, self.monitor)
            else:
                if self.verbose > 0:
                    logger.debug('Epoch %05d: saving model to %s', epoch, filepath)
                if self.save_weights_only:
                    self.model.layers[-(num_lambda_layers+1)].save_weights(filepath, overwrite=True)
                else:
                    self.model.layers[-(num_lambda_layers+1)].save(filepath, overwrite=True)
