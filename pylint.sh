set -e
echo 'Starting pylint checks'
pylint -d locally-disabled -f colorized src/main/python/deep_qa src/test/python/deep_qa_test src/main/python/*.py
echo -e "pylint checks passed\n"
