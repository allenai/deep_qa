# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.instance import TextInstance
from deep_qa.data.instances.text_encoders import text_encoders
from deep_qa.data.instances.true_false_instance import IndexedTrueFalseInstance, TrueFalseInstance


class TestTextInstance:
    """
    The point of this test class is to test the TextEncoder used by the TextInstance, to be sure
    that we get what we expect when using character encoders, or word-and-character encoders.
    """
    def test_words_tokenizes_the_sentence_correctly(self):
        t = TrueFalseInstance("This is a sentence.", None)
        assert t.words() == ['this', 'is', 'a', 'sentence', '.']
        TextInstance.encoder = text_encoders['characters']
        assert t.words() == ['T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 's', 'e', 'n', 't',
                             'e', 'n', 'c', 'e', '.']
        TextInstance.encoder = text_encoders['words and characters']
        assert t.words() == ['this', 'is', 'a', 'sentence', '.', 'T', 'h', 'i', 's', ' ', 'i', 's',
                             ' ', 'a', ' ', 's', 'e', 'n', 't', 'e', 'n', 'c', 'e', '.']
        TextInstance.encoder = text_encoders['word tokens']

    def test_to_indexed_instance_converts_correctly(self):
        data_indexer = DataIndexer()
        a_index = data_indexer.add_word_to_index("a")
        capital_a_index = data_indexer.add_word_to_index("A")
        space_index = data_indexer.add_word_to_index(" ")
        sentence_index = data_indexer.add_word_to_index("sentence")
        s_index = data_indexer.add_word_to_index("s")
        e_index = data_indexer.add_word_to_index("e")
        n_index = data_indexer.add_word_to_index("n")
        t_index = data_indexer.add_word_to_index("t")
        c_index = data_indexer.add_word_to_index("c")

        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [a_index, sentence_index]
        TextInstance.encoder = text_encoders['characters']
        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [capital_a_index, space_index, s_index, e_index, n_index, t_index,
                                         e_index, n_index, c_index, e_index]
        TextInstance.encoder = text_encoders['words and characters']
        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [[a_index, a_index],
                                         [sentence_index, s_index, e_index, n_index, t_index,
                                          e_index, n_index, c_index, e_index]]
        TextInstance.encoder = text_encoders['word tokens']


class TestIndexedInstance(TestCase):

    def test_get_lengths_works_with_words_and_characters(self):
        instance = IndexedTrueFalseInstance([[1, 2], [3, 1, 2]], True)
        assert instance.get_lengths() == {'word_sequence_length': 2, 'word_character_length': 3}

    def test_pad_word_sequence_handles_words_and_characters(self):
        instance = IndexedTrueFalseInstance([[1, 2], [3, 1, 2]], True)
        padded = instance.pad_word_sequence(instance.word_indices,
                                            {'word_sequence_length': 3, 'word_character_length': 4})
        assert padded == [[0, 0, 0, 0], [1, 2, 0, 0], [3, 1, 2, 0]]
