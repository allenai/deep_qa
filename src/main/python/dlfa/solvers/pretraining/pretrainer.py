class Pretrainer:
    """
    A pretrainer takes a solver, pulls out some of the layers, makes a new model, and trains it on
    some objective with some data.  Theoretically, you could train the whole model on the same
    objective as the solver does, but that would be "training", not "pretraining", so the idea here
    is that something is different from the main training step.

    A _really_ important point: any layer from the model that you pull out here has to be saved in
    the solver as a member variable!  If it is not, you could easily think you're pretraining a
    layer, but not actually changing the weights used in the solver itself, because the solver will
    just re-build that layer with new weights during the training step.
    """
    def __init__(self, solver):
        self.solver = solver

    def train():
        """
        Given some data and training parameters specified in constructor, run pre-training.  When
        this is done, the weights in the solver layers will have been updated during training, and
        you can just keep going with solver.train(), and things will just work.
        """
        raise NotImplementedError
