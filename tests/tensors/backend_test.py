# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_allclose
from keras import backend as K

from deep_qa.tensors.backend import cumulative_sum
from ..common.test_case import DeepQaTestCase


class TestBackendTensorFunctions(DeepQaTestCase):
    def test_cumulative_sum(self):
        vector = numpy.asarray([1, 2, 3, 4, 5])
        result = K.eval(cumulative_sum(K.variable(vector)))
        assert_allclose(result, [1, 3, 6, 10, 15])

        vector = numpy.asarray([[1, 2, 3, 4, 5],
                                [1, 1, 1, 1, 1],
                                [3, 5, 0, 0, 0]])
        result = K.eval(cumulative_sum(K.variable(vector)))
        assert_allclose(result, [[1, 3, 6, 10, 15],
                                 [1, 2, 3, 4, 5],
                                 [3, 8, 8, 8, 8]])
