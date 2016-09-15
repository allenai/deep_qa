
# run tests with specific backend, save exit codes so we can combine later
echo 'Python tests - THEANO BACKEND'
KERAS_BACKEND=theano py.test "$@"
THEANO_TEST=$?

echo 'Python tests - TENSORFLOW BACKEND'
KERAS_BACKEND=tensorflow py.test "$@"
TF_TEST=$?

# print individual backend test outcomes
echo 'Theano testing status:'
echo $THEANO_TEST
echo 'Tensorflow testing status:'
echo $TF_TEST

# return result - only exits with 0 if both tests pass.
exit $((TF_TEST + THEANO_TEST))
