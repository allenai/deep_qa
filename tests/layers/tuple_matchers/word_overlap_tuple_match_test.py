# pylint: disable=no-self-use
from unittest import TestCase

import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from numpy.testing import assert_array_almost_equal
from scipy.stats import logistic

from deep_qa.layers.tuple_matchers.word_overlap_tuple_matcher import WordOverlapTupleMatcher


class TestWordOverlapTupleMatch(TestCase):

    def setUp(self):
        num_slots = 3
        t1_words_per_slot = 4
        t2_words_per_slot = 3
        self.tuple1_input = Input(shape=(num_slots, t1_words_per_slot), dtype='int32', name="input_tuple1")
        self.tuple2_input = Input(shape=(num_slots, t2_words_per_slot), dtype='int32', name="input_tuple2")
        self.num_hidden_layers = 1
        self.hidden_layer_width = 2
        self.hidden_layer_activation = 'linear'

        self.tuple1 = np.array([[[1, 2, 0, 0],
                                 [1, 2, 3, 0],
                                 [0, 0, 0, 0]]])
        self.tuple2 = np.array([[[2, 1, 1],
                                 [1, 2, 0],
                                 [4, 5, 0]]])
        self.tuple3 = np.array([[[0, 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]]])


    def test_general_case(self):

        match_layer = WordOverlapTupleMatcher(self.num_hidden_layers, self.hidden_layer_width,
                                              hidden_layer_activation=self.hidden_layer_activation)
        output = match_layer([self.tuple1_input, self.tuple2_input])
        model = Model([self.tuple1_input, self.tuple2_input], output)

        # Get the initial weights for use in testing
        dense_hidden_weights = K.eval(model.trainable_weights[0])
        score_weights = K.eval(model.trainable_weights[1])

        # Testing general unmasked case.
        desired_overlap = np.array([[3/2,
                                     2/3,
                                     0]])
        # Desired_overlap gets fed into the inner NN.
        dense1_activation = np.dot(desired_overlap, dense_hidden_weights)
        final_score = np.dot(dense1_activation, score_weights)
        desired_result = logistic.cdf(final_score)
        print(desired_result)
        result = model.predict([self.tuple1, self.tuple2])
        assert_array_almost_equal(result, desired_result)

    def test_masks_handled_correctly(self):
        # Test when one tuple is all padding.
        # Here, since tuple3 is all padding, we want to return a mask value of 0 for this pair
        tuple1 = K.variable(self.tuple1)
        tuple3 = K.variable(self.tuple3)
        calculated_mask_exclude = K.eval(WordOverlapTupleMatcher(self.num_hidden_layers,
                                                                 self.hidden_layer_width,
                                                                  hidden_layer_activation=self.hidden_layer_activation)
                                         .compute_mask([tuple1, tuple3], [None, None]))
        assert_array_almost_equal(calculated_mask_exclude, np.array([0], dtype='int32'))

        # Test when tuple2 is valid.
        # Here, since tuple2 is valid, we want to return a mask value of 1 for this pair
        tuple2 = K.variable(self.tuple2)
        calculated_mask_include = K.eval(WordOverlapTupleMatcher(self.num_hidden_layers,
                                                                 self.hidden_layer_width,
                                                                 hidden_layer_activation=self.hidden_layer_activation)
                                         .compute_mask([tuple1, tuple2], [None, None]))
        assert_array_almost_equal(calculated_mask_include, np.array([1], dtype='int32'))
