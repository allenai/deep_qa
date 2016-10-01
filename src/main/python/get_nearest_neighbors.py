import argparse
import codecs
import logging

from deep_qa.solvers.differentiable_search import DifferentiableSearchSolver
from deep_qa.data.text_instance import TextInstance

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

def main():
    """
    This script loads a DifferentiableSearchSolver model, encodes a corpus and the sentences in a
    given file, and finds nearest neighbors in the corpus for all of the sentences in the file,
    using the trained sentence encoder.
    """
    argparser = argparse.ArgumentParser(description="Neural Network Solver")
    argparser.add_argument('--sentence_file', type=str, required=True,
                           help='Path to sentence file, for which we will find nearest neighbors')
    argparser.add_argument('--output_file', type=str, required=True,
                           help='Place to save results of nearest neighbor search')
    DifferentiableSearchSolver.update_arg_parser(argparser)

    args = argparser.parse_args()

    solver = DifferentiableSearchSolver(**vars(args))
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
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
