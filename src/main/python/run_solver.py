import logging
import sys

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
import numpy
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member

# pylint: disable=wrong-import-position

# Unless I find a better way to do this, we'll use ConfigFactory to read in a parameter file.
# ConfigFactory.parse_file() returns a ConfigTree, which is a subclass of OrderedDict, so the
# return value actually matches the type our code expects, anyway.
from pyhocon import ConfigFactory

from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.common.params import get_choice
from deep_qa.solvers import concrete_solvers

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    if len(sys.argv) != 2:
        print('USAGE: run_solver.py [param_file]')
        sys.exit(-1)

    param_file = sys.argv[1]
    params = ConfigFactory.parse_file(param_file)

    model_type = get_choice(params, 'solver_class', concrete_solvers.keys())
    solver_class = concrete_solvers[model_type]
    solver = solver_class(params)

    if solver.can_train():
        logger.info("Training model")
        solver.train()
    else:
        logger.info("Not enough training inputs.  Assuming you wanted to load a model instead.")
        # TODO(matt): figure out a way to specify which epoch you want to load a model from.
        solver.load_model()


if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
