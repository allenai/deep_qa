import logging
import sys

from deep_qa import run_model
from deep_qa.common.checks import ensure_pythonhashseed_set

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    if len(sys.argv) != 2:
        print('USAGE: run_model.py [param_file]')
        sys.exit(-1)

    run_model(sys.argv[1])

if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
