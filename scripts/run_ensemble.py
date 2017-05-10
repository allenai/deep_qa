import logging
import os
import sys

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deep_qa import score_dataset_with_ensemble, compute_accuracy
from deep_qa.common.checks import ensure_pythonhashseed_set

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    usage = 'USAGE: run_ensemble.py [param_file]+ -- [data_file]+'
    try:
        separator_index = sys.argv.index('--')
    except ValueError:
        print(usage)
        sys.exit(-1)
    param_files = sys.argv[1:separator_index]
    dataset_files = sys.argv[separator_index + 1:]
    predictions, labels = score_dataset_with_ensemble(param_files, dataset_files)
    compute_accuracy(predictions, labels)


if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
