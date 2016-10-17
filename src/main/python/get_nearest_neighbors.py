import argparse
import codecs
import logging

from pyhocon import ConfigFactory

from deep_qa.common.checks import ensure_pythonhashseed_set
from deep_qa.data.instances.instance import TextInstance
from deep_qa.solvers.with_memory.differentiable_search import DifferentiableSearchSolver

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    """
    This script loads a DifferentiableSearchSolver model, encodes a corpus and the sentences in a
    given file, and finds nearest neighbors in the corpus for all of the sentences in the file,
    using the trained sentence encoder.
    """
    argparser = argparse.ArgumentParser(description="Neural Network Solver")
    argparser.add_argument('--param_file', type=str, required=True,
                           help='Path to file containing solver parameters')
    argparser.add_argument('--sentence_file', type=str, required=True,
                           help='Path to sentence file, for which we will find nearest neighbors')
    argparser.add_argument('--output_file', type=str, required=True,
                           help='Place to save results of nearest neighbor search')
    args = argparser.parse_args()

    param_file = args.param_file
    params = ConfigFactory.parse_file(param_file)
    solver = DifferentiableSearchSolver(**params)  # TODO(matt): fix this in the next PR
    solver.load_model()

    with codecs.open(args.output_file, 'w', 'utf-8') as outfile:
        for line in codecs.open(args.sentence_file, 'r', 'utf-8').readlines():
            outfile.write(line)
            instance = TextInstance(line.strip(), True)
            neighbors = solver.get_nearest_neighbors(instance)
            for neighbor in neighbors:
                outfile.write('\t')
                outfile.write(neighbor.text)
                outfile.write('\n')
            outfile.write('\n')


if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
