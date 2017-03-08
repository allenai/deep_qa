# pylint: disable=no-self-use,invalid-name
import numpy

from keras.layers import Input

from deep_qa.layers.encoders import BOWEncoder
from deep_qa.layers.time_distributed_embedding import TimeDistributedEmbedding
from deep_qa.layers.wrappers.encoder_wrapper import EncoderWrapper
from deep_qa.layers.wrappers.time_distributed_with_mask import TimeDistributedWithMask
from deep_qa.layers.wrappers.output_mask import OutputMask
from deep_qa.layers.tuple_matchers.slot_similarity_tuple_matcher import SlotSimilarityTupleMatcher
from deep_qa.layers.tuple_matchers.word_overlap_tuple_matcher import WordOverlapTupleMatcher
from deep_qa.training.models import DeepQaModel
from ...common.test_case import DeepQaTestCase

class TestTimeDistributedWithMask(DeepQaTestCase):
    def test_handles_multiple_masks(self):
        # We'll use the SlotSimilarityTupleMatcher to test this, because it takes two masked
        # inputs.  Here we're using an input of shape (batch_size, num_options, num_tuples,
        # num_slots, num_words).
        tuple_input = Input(shape=(2, 3, 4, 5), dtype='int32')
        tuple_input_2 = Input(shape=(2, 3, 4, 5), dtype='int32')
        embedding = TimeDistributedEmbedding(input_dim=3, output_dim=6, mask_zero=True)
        # shape is now (batch_size, num_options, num_tuples, num_slots, num_words, embedding_dim)
        embedded_tuple = embedding(tuple_input)
        embedded_tuple_2 = embedding(tuple_input_2)
        encoder = EncoderWrapper(EncoderWrapper(EncoderWrapper(BOWEncoder())))
        # shape is now (batch_size, num_options, num_tuples, num_slots, embedding_dim)
        encoded_tuple = encoder(embedded_tuple)
        encoded_tuple_2 = encoder(embedded_tuple_2)
        # Shape of input to the tuple matcher is [(batch size, 2, 3, 4, 6), (batch size, 2, 3, 4, 6)]
        # Shape of input_mask to the tuple matcher is  [(batch size, 2, 3, 4), (batch size, 2, 3, 4)]
        # Expected output mask shape (batch_size, 2, 3)
        time_distributed = TimeDistributedWithMask(TimeDistributedWithMask(
                SlotSimilarityTupleMatcher({"type": "cosine_similarity"})))

        time_distributed_output = time_distributed([encoded_tuple, encoded_tuple_2])
        mask_output = OutputMask()(time_distributed_output)
        model = DeepQaModel(input=[tuple_input, tuple_input_2], output=mask_output)
        zeros = [0, 0, 0, 0, 0]
        non_zeros = [1, 1, 1, 1, 1]
        # shape: (batch size, num_options, num_tuples, num_slots, num_words), or (1, 2, 3, 4, 5)
        tuples1 = numpy.asarray([[[[zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [non_zeros, non_zeros, zeros, zeros]],
                                  [[non_zeros, non_zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [zeros, zeros, zeros, zeros]]]])
        tuples2 = numpy.asarray([[[[non_zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [zeros, zeros, zeros, zeros]],
                                  [[non_zeros, non_zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros]]]])
        actual_mask = model.predict([tuples1, tuples2])
        expected_mask = numpy.asarray([[[0, 1, 0], [1, 1, 0]]]) # shape: (batch size, num_options, num_tuples)
        assert actual_mask.shape == (1, 2, 3)
        numpy.testing.assert_array_almost_equal(expected_mask, actual_mask)

    def test_returns_masks_if_no_input_mask(self):
        # We'll use the SlotSimilarityTupleMatcher to test this, because it takes two masked
        # inputs.  Here we're using an input of shape (batch_size, num_options, num_tuples,
        # num_slots, num_words).
        tuple_input = Input(shape=(2, 3, 4, 5), dtype='int32')
        tuple_input_2 = Input(shape=(2, 3, 4, 5), dtype='int32')
        # shape is (batch_size, num_options, num_tuples, num_slots, num_words)

        # Shape of input to the tuple matcher is [(batch size, 2, 3, 4, 5), (batch size, 2, 3, 4, 5)]
        # Shape of input_mask to the tuple matcher is  [None, None]
        # Expected output mask shape (batch_size, 2, 3)
        time_distributed = TimeDistributedWithMask(TimeDistributedWithMask(
                WordOverlapTupleMatcher()))

        time_distributed_output = time_distributed([tuple_input, tuple_input_2])
        mask_output = OutputMask()(time_distributed_output)
        model = DeepQaModel(input=[tuple_input, tuple_input_2], output=mask_output)
        zeros = [0, 0, 0, 0, 0]
        non_zeros = [1, 1, 1, 1, 1]
        # shape: (batch size, num_options, num_tuples, num_slots, num_words), or (1, 2, 3, 4, 5)
        tuples1 = numpy.asarray([[[[zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [non_zeros, non_zeros, zeros, zeros]],
                                  [[non_zeros, non_zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [zeros, zeros, zeros, zeros]]]])
        tuples2 = numpy.asarray([[[[non_zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [zeros, zeros, zeros, zeros]],
                                  [[non_zeros, non_zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros],
                                   [non_zeros, zeros, zeros, zeros]]]])
        actual_mask = model.predict([tuples1, tuples2])
        expected_mask = numpy.asarray([[[0, 1, 0], [1, 1, 0]]]) # shape: (batch size, num_options, num_tuples)
        assert actual_mask.shape == (1, 2, 3)
        numpy.testing.assert_array_almost_equal(expected_mask, actual_mask)
