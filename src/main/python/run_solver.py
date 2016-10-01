import argparse
import logging
import sys

# These have to be before we do any import from keras.  It would be nice to be able to pass in a
# value for this, but that makes argument passing a whole lot more complicated.  If/when we change
# how arguments work (e.g., loading a file), then we can think about setting this as a parameter.
import random
import numpy
random.seed(13370)
numpy.random.seed(1337)  # pylint: disable=no-member

from deep_qa.solvers import concrete_solvers  # pylint: disable=wrong-import-position

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    # The first argument to this script must be a model type.  We will use that model type to
    # construct an argument parser, which will only allow arguments relevant to the model type.
    model_type = sys.argv[1]
    if model_type not in concrete_solvers:
        print("First argument must be model class, one of", concrete_solvers.keys())
        sys.exit(-1)
    solver_class = concrete_solvers[model_type]

    argparser = argparse.ArgumentParser(description="Neural Network Solver")
    solver_class.update_arg_parser(argparser)

    # We already handled the first argument, so we just parse the remaining ones.
    args = argparser.parse_args(sys.argv[2:])

    # The solver class got to update the argument parser, so it will just grab whatever arguments
    # it needs directly from the parsed arguments.
    solver = solver_class(**vars(args))

    if solver.can_train():
        logger.info("Training model")
        solver.train()
    else:
        logger.info("Not enough training inputs.  Assuming you wanted to load a model instead.")
        solver.load_model(args.use_model_from_epoch)

    if solver.can_test():
        solver.test()


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
