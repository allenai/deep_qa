import keras.backend as K
import pytest

# pylint: disable=invalid-name,protected-access

backend = K._backend

# These are decorators for tests(classes and methods) which exclude a test if the condition is
# satisfied.  They need to live here so that they can be imported in the test directory universally
# and they require the python_path in pytest.ini to include this directory.

# Eg:

# from test_markers import req_tensorflow(in a test directory)
#
#     @requires_tensorflow
#     def test_some_cool_tensorflow_thing(args):
#
#         test_stuff....
#
# Would only run if the backend of Keras is tensorflow.  Pytest will show this as SKIPPED in the
# test report.

requires_tensorflow = pytest.mark.skipif(
        backend == "theano",
        reason="Current Keras backend = theano. This test is for Tensorflow only."
        )

requires_theano = pytest.mark.skipif(
        backend == "tensorflow",
        reason="Current Keras backend = tensorflow. This test is for Theano only."
        )
