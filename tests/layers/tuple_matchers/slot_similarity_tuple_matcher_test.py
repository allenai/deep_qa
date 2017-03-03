# pylint: disable=no-self-use
import numpy
from numpy.testing import assert_array_almost_equal
from keras import backend as K
from keras.layers import Input, Masking
from keras.models import Model
from scipy.stats import logistic

from deep_qa.layers.tuple_matchers.slot_similarity_tuple_matcher import SlotSimilarityTupleMatcher
from deep_qa.tensors.similarity_functions.cosine_similarity import CosineSimilarity
from ...common.test_case import DeepQaTestCase


class TestSlotSimilarityTupleMatcher(DeepQaTestCase):
    def setUp(self):
        super(TestSlotSimilarityTupleMatcher, self).setUp()
        num_slots = 3
        embed_dimension = 4
        self.tuple1_input = Input(shape=(num_slots, embed_dimension), dtype='float32', name="input_tuple1")
        self.tuple2_input = Input(shape=(num_slots, embed_dimension), dtype='float32', name="input_tuple2")
        self.num_hidden_layers = 1
        self.hidden_layer_width = 2
        self.hidden_layer_activation = 'linear'

        # tuple1 shape: (batch size, num_slots, embed_dimension)
        self.tuple1 = numpy.random.rand(1, num_slots, embed_dimension)
        # tuple2 shape: (batch size, num_slots, embed_dimension)
        self.tuple2 = numpy.random.rand(1, num_slots, embed_dimension)

    def test_general_case(self):

        match_layer = SlotSimilarityTupleMatcher({"type": "cosine_similarity"},
                                                 self.num_hidden_layers,
                                                 self.hidden_layer_width,
                                                 hidden_layer_activation=self.hidden_layer_activation)
        output = match_layer([self.tuple1_input, self.tuple2_input])
        model = Model([self.tuple1_input, self.tuple2_input], output)

        # Get the initial weights for use in testing
        dense_hidden_weights = K.eval(model.trainable_weights[0])
        score_weights = K.eval(model.trainable_weights[1])

        # Testing general unmasked case.
        similarity_function = CosineSimilarity(name="cosine_similarity")
        cosine_similarities = similarity_function.compute_similarity(K.variable(self.tuple1),
                                                                     K.variable(self.tuple2))

        # Desired_overlap gets fed into the inner NN.
        dense1_activation = numpy.dot(K.eval(cosine_similarities), dense_hidden_weights)
        final_score = numpy.dot(dense1_activation, score_weights)
        # Apply the final sigmoid activation function.
        desired_result = logistic.cdf(final_score)
        result = model.predict([self.tuple1, self.tuple2])
        assert_array_almost_equal(result, desired_result)

    def test_returns_masks_correctly(self):
        # Test when one tuple is all padding.
        # Here, since tuple3 is all padding, we want to return a mask value of 0 for this pair
        tuple1 = K.variable(self.tuple1)
        mask1 = K.variable(numpy.asarray([[1, 1, 1]]))
        tuple2 = K.variable(self.tuple2)
        mask2 = K.variable(numpy.asarray([[0, 0, 0]]))
        calculated_mask_exclude = K.eval(
                SlotSimilarityTupleMatcher({"type": "cosine_similarity"},
                                           self.num_hidden_layers,
                                           self.hidden_layer_width,
                                           hidden_layer_activation=self.hidden_layer_activation)
                .compute_mask([tuple1, tuple2], [mask1, mask2]))
        assert_array_almost_equal(calculated_mask_exclude, numpy.array([[0]], dtype='int32'))
        assert calculated_mask_exclude.shape == (1, 1,)

        # Test when tuple2 is valid.
        # Here, since tuple2 is valid, we want to return a mask value of 1 for this pair
        mask2 = K.variable(numpy.asarray([[0, 1, 0]]))
        calculated_mask_include = K.eval(
                SlotSimilarityTupleMatcher({"type": "cosine_similarity"},
                                           self.num_hidden_layers,
                                           self.hidden_layer_width,
                                           hidden_layer_activation=self.hidden_layer_activation)
                .compute_mask([tuple1, tuple2], [mask1, mask2]))
        assert_array_almost_equal(calculated_mask_include, numpy.array([[1]], dtype='int32'))
        assert calculated_mask_include.shape == (1, 1,)

    def test_handles_input_masks_correctly(self):
        mask_layer = Masking(mask_value=0.0)
        masked_tuple1 = mask_layer(self.tuple1_input)
        masked_tuple2 = mask_layer(self.tuple2_input)
        # Add a set of paddings to slot 1 in tuple 2
        self.tuple2[:, 1, :] = numpy.zeros(4)
        match_layer = SlotSimilarityTupleMatcher({"type": "cosine_similarity"},
                                                 self.num_hidden_layers,
                                                 self.hidden_layer_width,
                                                 hidden_layer_activation=self.hidden_layer_activation)
        output = match_layer([masked_tuple1, masked_tuple2])
        mask_model = Model([self.tuple1_input, self.tuple2_input], output)

        similarity_function = CosineSimilarity(name="cosine_similarity")
        cosine_similarities = similarity_function.compute_similarity(K.variable(self.tuple1),
                                                                     K.variable(self.tuple2))
        mask2 = K.variable(numpy.asarray([[1, 0, 1]], dtype='float32'))
        masked_cosine_similarities = cosine_similarities * mask2

        # Get the initial weights for use in testing
        dense_hidden_weights = K.eval(mask_model.trainable_weights[0])
        score_weights = K.eval(mask_model.trainable_weights[1])

        # Desired_overlap gets fed into the inner NN.
        dense1_activation = numpy.dot(K.eval(masked_cosine_similarities), dense_hidden_weights)
        final_score = numpy.dot(dense1_activation, score_weights)
        # Apply the final sigmoid activation function.
        desired_result = logistic.cdf(final_score)
        result = mask_model.predict([self.tuple1, self.tuple2])
        assert_array_almost_equal(result, desired_result)
