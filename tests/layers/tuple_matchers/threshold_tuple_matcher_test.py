# pylint: disable=no-self-use
import numpy
from numpy.testing import assert_array_almost_equal
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras import activations

from deep_qa.tensors.backend import apply_feed_forward
from deep_qa.layers.time_distributed_embedding import TimeDistributedEmbedding
from deep_qa.layers.tuple_matchers.threshold_tuple_matcher import ThresholdTupleMatcher
from ...common.test_case import DeepQaTestCase


class TestSlotSimilarityTupleMatcher(DeepQaTestCase):
    def setUp(self):
        super(TestSlotSimilarityTupleMatcher, self).setUp()
        self.num_slots = 3
        self.num_words = 5
        self.embed_dimension = 4
        self.tuple1_input = Input(shape=(self.num_slots, self.num_words, self.embed_dimension), dtype='float32',
                                  name="input_tuple1")
        self.tuple2_input = Input(shape=(self.num_slots, self.num_words, self.embed_dimension), dtype='float32',
                                  name="input_tuple2")
        self.num_hidden_layers = 1
        self.hidden_layer_width = 2
        self.hidden_layer_activation = 'linear'

        # tuple1 shape: (batch size, num_slots, num_words, embed_dimension)
        self.tuple1 = numpy.random.rand(1, self.num_slots, self.num_words, self.embed_dimension)
        # tuple2 shape: (batch size, num_slots, num_words, embed_dimension)
        self.tuple2 = numpy.random.rand(1, self.num_slots, self.num_words, self.embed_dimension)
        # cause some dimensions to have identical embeddings (i.e. cosine similarity of 1)
        self.tuple1[0, 1, 1, :] = numpy.ones(4)
        self.tuple1[0, 2, 0, :] = numpy.ones(4)

        self.tuple2[0, 1, 0, :] = numpy.ones(4)
        self.tuple2[0, 2, 3, :] = numpy.ones(4)
        self.tuple2[0, 2, 4, :] = numpy.ones(4)
        # This should result in 1 matched word in slot 1, and two in slot 2
        # So the normalized overlaps should be (0, 1/5,  2/5)

    def test_general_case(self):
        # Custom initialization for this test to account for rounding in the cosine similarity.
        # pylint: disable=unused-argument
        def custom_initialization(shape, name=None):
            return K.ones(shape) * 0.999

        match_layer = ThresholdTupleMatcher({"type": "cosine_similarity"},
                                            self.num_hidden_layers,
                                            self.hidden_layer_width,
                                            initialization=custom_initialization,
                                            hidden_layer_activation=self.hidden_layer_activation)
        output = match_layer([self.tuple1_input, self.tuple2_input])
        model = Model([self.tuple1_input, self.tuple2_input], output)

        # Get the initial weights for use in testing
        layer_nn = match_layer.hidden_layer_weights

        # Testing general unmasked case.
        desired_overlap = K.variable(numpy.asarray([[0, 1/5, 2/5]]))
        # Desired_overlap gets fed into the inner NN.
        neural_network_feed_forward = apply_feed_forward(desired_overlap, layer_nn,
                                                         activations.get(match_layer.hidden_layer_activation))
        # Apply the final activation function.
        desired_result = activations.get(match_layer.final_activation)(K.dot(neural_network_feed_forward,
                                                                             match_layer.score_layer))
        result = model.predict([self.tuple1, self.tuple2])
        assert_array_almost_equal(result, K.eval(desired_result))


    def test_returns_masks_correctly(self):
        # Test when one tuple is all padding.
        # Here, since tuple2 is all padding, we want to return a mask value of 0 for this pair
        # tuple1 shape: (batch size, num_slots, num_words, embed_dimension)
        tuple1 = K.variable(self.tuple1)
        mask1 = K.variable(numpy.ones((1, self.num_slots, self.num_words)))
        tuple2 = K.variable(self.tuple2)
        mask2 = K.variable(numpy.zeros((1, self.num_slots, self.num_words)))
        calculated_mask_exclude = K.eval(
                ThresholdTupleMatcher({"type": "cosine_similarity"},
                                      self.num_hidden_layers,
                                      self.hidden_layer_width,
                                      hidden_layer_activation=self.hidden_layer_activation)
                .compute_mask([tuple1, tuple2], [mask1, mask2]))
        assert_array_almost_equal(calculated_mask_exclude, numpy.array([[0]], dtype='int32'))
        assert calculated_mask_exclude.shape == (1, 1,)

        # Test when tuple2 is valid.
        # Here, since tuple2 is valid, we want to return a mask value of 1 for this pair
        new_mask = numpy.ones((1, self.num_slots, self.num_words))
        new_mask[:, :, 1] = 0
        mask2 = K.variable(new_mask)

        calculated_mask_include = K.eval(
                ThresholdTupleMatcher({"type": "cosine_similarity"},
                                      self.num_hidden_layers,
                                      self.hidden_layer_width,
                                      hidden_layer_activation=self.hidden_layer_activation)
                .compute_mask([tuple1, tuple2], [mask1, mask2]))
        assert_array_almost_equal(calculated_mask_include, numpy.array([[1]], dtype='int32'))
        assert calculated_mask_include.shape == (1, 1,)

    def test_handles_input_masks_correctly(self):
        # Custom initialization for this test to account for rounding in the cosine similarity.
        # pylint: disable=unused-variable,unused-argument
        def custom_initialization(shape, name=None):
            return K.ones(shape) * 0.999

        num_slots = 3
        num_words = 5
        embed_dimension = 4
        tuple1_word_input = Input(shape=(num_slots, num_words), dtype='int32', name="input_tuple1")
        tuple2_word_input = Input(shape=(num_slots, num_words), dtype='int32', name="input_tuple2")

        embedding = TimeDistributedEmbedding(10, embed_dimension, mask_zero=True)
        embedded_masked_tuple1 = embedding(tuple1_word_input)
        embedded_masked_tuple2 = embedding(tuple2_word_input)

        match_layer = ThresholdTupleMatcher({"type": "cosine_similarity"},
                                            self.num_hidden_layers,
                                            self.hidden_layer_width,
                                            initialization=custom_initialization,
                                            hidden_layer_activation=self.hidden_layer_activation)
        output = match_layer([embedded_masked_tuple1, embedded_masked_tuple2])
        mask_model = Model([tuple1_word_input, tuple2_word_input], output)

        # Assign tuple1 to be all 4's and tuple2 to be all 3's so we can control lexical overlap
        tuple1_words = numpy.ones((1, num_slots, num_words)) * 4
        tuple2_words = numpy.ones((1, num_slots, num_words)) * 3
        # Add a set of matching zeros to slot 1 in each tuple1 -- but shouldn't "match" because it's padding
        tuple1_words[:, 1, :] = numpy.zeros(num_words)
        tuple2_words[:, 1, :] = numpy.zeros(num_words)

        # Get the initial weights for use in testing
        layer_nn = match_layer.hidden_layer_weights

        # Testing general unmasked case.
        desired_overlap = K.variable(numpy.asarray([[0, 0, 0]]))
        # Desired_overlap gets fed into the inner NN.
        neural_network_feed_forward = apply_feed_forward(desired_overlap, layer_nn,
                                                         activations.get(match_layer.hidden_layer_activation))
        # Apply the final activation function.
        desired_result = activations.get(match_layer.final_activation)(K.dot(neural_network_feed_forward,
                                                                             match_layer.score_layer))
        result = mask_model.predict([tuple1_words, tuple2_words])
        assert_array_almost_equal(result, K.eval(desired_result))
