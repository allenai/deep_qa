# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers import TimeDistributedEmbedding, VectorMatrixSplit
from deep_qa.layers.wrappers import OutputMask

class TestVectorMatrixSplit:
    def test_split_works_correctly_on_word_indices(self):
        vocabulary_size = 10
        sentence_length = 10
        word_length = 5
        num_sentences = 7
        sentence_input = Input(shape=(sentence_length, word_length), dtype='int32')
        split_layer = VectorMatrixSplit(split_axis=2)
        words, characters = split_layer(sentence_input)
        model = Model(inputs=[sentence_input], outputs=[words, characters])
        sentence_tensor = numpy.random.randint(0, vocabulary_size, (num_sentences, sentence_length, word_length))
        word_tensor, character_tensor = model.predict([sentence_tensor])
        assert numpy.array_equal(word_tensor, sentence_tensor[:, :, 0])
        assert numpy.array_equal(character_tensor, sentence_tensor[:, :, 1:])

    def test_split_works_correctly_with_negative_axis(self):
        vocabulary_size = 10
        sentence_length = 10
        word_length = 5
        num_sentences = 7
        sentence_input = Input(shape=(sentence_length, word_length), dtype='int32')
        split_layer = VectorMatrixSplit(split_axis=-1)
        words, characters = split_layer(sentence_input)
        model = Model(inputs=[sentence_input], outputs=[words, characters])
        sentence_tensor = numpy.random.randint(0, vocabulary_size, (num_sentences, sentence_length, word_length))
        word_tensor, character_tensor = model.predict([sentence_tensor])
        assert numpy.array_equal(word_tensor, sentence_tensor[:, :, 0])
        assert numpy.array_equal(character_tensor, sentence_tensor[:, :, 1:])

    def test_split_works_correctly_on_word_embeddings_with_masking(self):
        vocabulary_size = 10
        sentence_length = 10
        word_length = 5
        embedding_dim = 10
        num_sentences = 7
        sentence_input = Input(shape=(sentence_length, word_length), dtype='int32')
        embedding = TimeDistributedEmbedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        embedded_sentence = embedding(sentence_input)  # (batch_size, sentence_length, word_length, embedding_dim)
        sentence_mask = OutputMask()(embedded_sentence)
        # Note that this mask_split_axis doesn't make practical sense; I'm just testing the code
        # with a different axis for the mask and the input.
        split_layer = VectorMatrixSplit(split_axis=2, mask_split_axis=1)
        words, characters = split_layer(embedded_sentence)
        word_mask = OutputMask()(words)
        character_mask = OutputMask()(characters)
        outputs = [embedded_sentence, words, characters, sentence_mask, word_mask, character_mask]
        model = Model(inputs=[sentence_input], outputs=outputs)
        sentence_tensor = numpy.random.randint(0, vocabulary_size, (num_sentences, sentence_length, word_length))
        actual_outputs = model.predict([sentence_tensor])
        sentence_tensor, word_tensor, character_tensor, sentence_mask, word_mask, character_mask = actual_outputs
        assert numpy.array_equal(word_tensor, sentence_tensor[:, :, 0, :])
        assert numpy.array_equal(character_tensor, sentence_tensor[:, :, 1:, :])
        assert numpy.array_equal(word_mask, sentence_mask[:, 0, :])
        assert numpy.array_equal(character_mask, sentence_mask[:, 1:, :])
