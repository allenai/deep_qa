set -e
echo 'Starting pylint checks'
pylint -d locally-disabled,locally-enabled -f colorized deep_qa tests scripts/*.py
echo -e "pylint checks passed\n"
