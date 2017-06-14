# pylint: disable=no-self-use,invalid-name
import numpy
from numpy.testing import assert_array_equal

from keras.layers import Embedding, Input
from keras.models import Model

from deep_qa.layers import VectorMatrixMerge
from deep_qa.layers.wrappers import OutputMask

class TestVectorMatrixMerge:
    def test_merge_works_correctly_on_word_indices(self):
        vocab_size = 10
        sentence_length = 10
        word_length = 7
        num_sentences = 7
        for concat_axis in [2, -1]:
            sentence_input = Input(shape=(sentence_length, word_length - 2), dtype='int32')
            additional_input = Input(shape=(sentence_length,), dtype='int32')
            additional_input2 = Input(shape=(sentence_length,), dtype='int32')
            merge_layer = VectorMatrixMerge(concat_axis=concat_axis)
            merged = merge_layer([additional_input, additional_input2, sentence_input])
            model = Model(inputs=[sentence_input, additional_input, additional_input2], outputs=merged)
            sentence_tensor = numpy.random.randint(0, vocab_size, (num_sentences, sentence_length, word_length))
            merged_tensor = model.predict([sentence_tensor[:, :, 2:],
                                           sentence_tensor[:, :, 0],
                                           sentence_tensor[:, :, 1]])
            assert_array_equal(sentence_tensor, merged_tensor)

    def test_merge_adds_words_to_sentence_correctly(self):
        # The thing to note here is that when we're adding words, we're adding rows to the mask as
        # well.  This test makes sure that this works correctly.
        vocab_size = 10
        sentence_length = 3
        word_length = 3
        embedding_dim = 10
        sentence_input = Input(shape=(sentence_length, word_length), dtype='int32')
        extra_word_input = Input(shape=(word_length,), dtype='int32')
        embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
        embedded_sentence = embedding(sentence_input)  # (batch_size, sentence_length, word_length, embedding_dim)
        embedded_extra_word = embedding(extra_word_input)  # (batch_size, word_length, embedding_dim)
        merge_layer = VectorMatrixMerge(concat_axis=1)
        merged_sentence = merge_layer([embedded_extra_word, embedded_sentence])
        result_mask = OutputMask()(merged_sentence)
        model = Model(inputs=[sentence_input, extra_word_input], outputs=[merged_sentence, result_mask])
        sentence_tensor = numpy.asarray([[[1, 3, 0], [2, 8, 7], [0, 0, 0]]])
        extra_word_tensor = numpy.asarray([[9, 0, 0]])
        merged_tensor, result_mask_tensor = model.predict([sentence_tensor, extra_word_tensor])
        expected_mask = numpy.asarray([[[1, 0, 0], [1, 1, 0], [1, 1, 1], [0, 0, 0]]])
        assert merged_tensor.shape == (1, sentence_length + 1, word_length, embedding_dim)
        assert_array_equal(result_mask_tensor, expected_mask)

    def test_merge_adds_dims_to_word_embedding_correctly(self):
        # The thing to note here is that when we're adding dimensions to an embedding, we're not
        # changing the mask.  That is, the concat axis is greater than the dimensionality of the
        # mask.  This test makes sure that this works correctly.
        vocab_size = 10
        sentence_length = 6
        embedding_dim = 10
        for concat_axis in [2, -1]:
            sentence_input = Input(shape=(sentence_length,), dtype='int32')
            extra_embedding_input = Input(shape=(sentence_length,), dtype='float32')
            embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)
            embedded_sentence = embedding(sentence_input)  # (batch_size, sentence_length, embedding_dim)
            merge_layer = VectorMatrixMerge(concat_axis=concat_axis)
            merged_sentence = merge_layer([extra_embedding_input, embedded_sentence])
            result_mask = OutputMask()(merged_sentence)
            model = Model(inputs=[sentence_input, extra_embedding_input], outputs=[merged_sentence, result_mask])
            sentence_tensor = numpy.asarray([[1, 3, 6, 2, 0, 0]])
            extra_word_tensor = numpy.asarray([[1, 2, 3, 4, 5, 6]])
            merged_tensor, result_mask_tensor = model.predict([sentence_tensor, extra_word_tensor])
            expected_mask = numpy.asarray([[1, 1, 1, 1, 0, 0]])
            assert merged_tensor.shape == (1, sentence_length, embedding_dim + 1)
            assert_array_equal(merged_tensor[0, :, 0], [1, 2, 3, 4, 5, 6])
            assert_array_equal(result_mask_tensor, expected_mask)
