from typing import Any, Dict
from overrides import overrides

from ...data.instance import Instance
from ..trainer import Trainer

class Pretrainer(Trainer):
    # pylint: disable=abstract-method
    """
    A Pretrainer is a Trainer that takes another Trainer as input, and trains (part of) that
    Trainer's model on some other data, likely using some other objective.  Theoretically, you
    could train the whole model on the same objective as the Trainer does, but that would be
    "training", not "pretraining", so the idea here is that something is different from the main
    training step.

    A _really_ important point: any layer from the model that you pull out here has to be saved in
    the Trainer as a member variable!  If it is not, you could easily think you're pretraining a
    layer, but not actually changing the weights used in the trainer itself, because the Trainer
    will just re-build that layer with new weights during its training step.
    """
    # While it's not great, we need access to a few of the internals of the other Trainer, so we'll
    # disable protected access checks.
    # pylint: disable=protected-access
    def __init__(self, trainer: Trainer, params: Dict[str, Any]):
        # The default for saving models in Trainer is True.  We probably don't want to save the
        # models from the pretrainer, so we'll change the default.
        if 'save_models' not in params:
            params['save_models'] = False
        super(Pretrainer, self).__init__(params)
        self.trainer = trainer

    def on_finished(self):
        """
        This method is called when all of the pre-training is finished.  You can do whatever clean
        up or whatever you need to here.  Default implementation is a no-op.
        """
        pass

    @overrides
    def _prepare_instance(self, instance: Instance, make_batch: bool=True):
        """
        This method is only called for test-time predictions.  It really seems unlikely that you'd
        want to use this for a pre-trainer, so we override it here to get rid of the
        NotImplementedError.  You can override it yourself with a real implementation if you really
        want to use this somewhere.
        """
        raise RuntimeError("Why are you preparing an instance on a pretrainer?")
