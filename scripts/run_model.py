import logging
import os
import sys

# pylint: disable=wrong-import-position
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from deep_qa import run_model, evaluate_model
from deep_qa.common.checks import ensure_pythonhashseed_set

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def main():
    usage = 'USAGE: run_model.py [param_file] [train|test]'
    if len(sys.argv) == 2:
        run_model(sys.argv[1])
    elif len(sys.argv) == 3:
        mode = sys.argv[2]
        if mode == 'train':
            run_model(sys.argv[1])
        elif mode == 'test':
            evaluate_model(sys.argv[1])
        else:
            print(usage)
            sys.exit(-1)
    else:
        print(usage)
        sys.exit(-1)

if __name__ == "__main__":
    ensure_pythonhashseed_set()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
