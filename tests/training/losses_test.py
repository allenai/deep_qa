# pylint: disable=invalid-name,no-self-use
from keras import backend as K
import numpy
from numpy.testing import assert_almost_equal

from deep_qa.training.losses import ranking_loss, ranking_loss_with_margin

from ..common.test_case import DeepQaTestCase


class TestLosses(DeepQaTestCase):
    def test_ranking_loss_is_computed_correctly(self):
        predictions = numpy.asarray([[.1, .4, .8], [-.1, -.2, .1]])
        labels = numpy.asarray([[0, 0, 1], [1, 0, 0]])
        sigmoid = lambda x: 1.0 / (1.0 + numpy.exp(-x))
        expected_result = numpy.mean(-sigmoid(numpy.asarray([.8 - .4, -.1 - .1])))
        result = K.eval(ranking_loss(K.variable(predictions), K.variable(labels)))
        assert_almost_equal(expected_result, result)

    def test_ranking_loss_with_margin_is_computed_correctly(self):
        predictions = numpy.asarray([[.1, .4, .8], [-.1, -.2, .1]])
        labels = numpy.asarray([[0, 0, 1], [1, 0, 0]])
        expected_result = numpy.mean(numpy.maximum(0, numpy.asarray([1 + .4 - .8, 1 + .1 - -.1])))
        result = K.eval(ranking_loss_with_margin(K.variable(predictions), K.variable(labels)))
        assert_almost_equal(expected_result, result)
