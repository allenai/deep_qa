# pylint: disable=no-self-use,invalid-name
from deep_qa.common.params import Params
from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances import TextInstance
from deep_qa.data.instances.language_modeling import IndexedSentenceInstance
from deep_qa.data.instances.language_modeling import SentenceInstance
from deep_qa.data.tokenizers import tokenizers
from deep_qa.testing.test_case import DeepQaTestCase
from numpy.testing import assert_array_equal


class TestSentenceInstance(DeepQaTestCase):
    def setUp(self):
        super(TestSentenceInstance, self).setUp()
        self.data_indexer = DataIndexer()
        self.this_index = self.data_indexer.add_word_to_index("this")
        self.is_index = self.data_indexer.add_word_to_index("is")
        self.a_index = self.data_indexer.add_word_to_index("a")
        self.sentence_index = self.data_indexer.add_word_to_index("sentence")
        self.start_index = self.data_indexer.add_word_to_index("<S>")
        self.end_index = self.data_indexer.add_word_to_index("</S>")
        self.space_index = self.data_indexer.add_word_to_index(' ')
        self.c_index = self.data_indexer.add_word_to_index('c')
        self.e_index = self.data_indexer.add_word_to_index('e')
        self.h_index = self.data_indexer.add_word_to_index('h')
        self.i_index = self.data_indexer.add_word_to_index('i')
        self.n_index = self.data_indexer.add_word_to_index('n')
        self.s_index = self.data_indexer.add_word_to_index('s')
        self.t_index = self.data_indexer.add_word_to_index('t')
        self.a_char_index = self.data_indexer.add_word_to_index('a', namespace='characters')
        self.c_char_index = self.data_indexer.add_word_to_index('c', namespace='characters')
        self.e_char_index = self.data_indexer.add_word_to_index('e', namespace='characters')
        self.h_char_index = self.data_indexer.add_word_to_index('h', namespace='characters')
        self.i_char_index = self.data_indexer.add_word_to_index('i', namespace='characters')
        self.n_char_index = self.data_indexer.add_word_to_index('n', namespace='characters')
        self.s_char_index = self.data_indexer.add_word_to_index('s', namespace='characters')
        self.t_char_index = self.data_indexer.add_word_to_index('t', namespace='characters')

    def tearDown(self):
        super(TestSentenceInstance, self).tearDown()
        TextInstance.tokenizer = tokenizers['words'](Params({}))

    @staticmethod
    def instance_to_line(text, index=None):
        index_str = '' if index is None else str(index) + '\t'
        return index_str + text

    def test_read_from_line_handles_one_column(self):
        text = "this is a sentence"
        instance = SentenceInstance.read_from_line(text)
        assert instance.text == text
        assert instance.label is None
        assert instance.index is None

    def test_read_from_line_handles_two_column(self):
        index = 23
        text = "this is a sentence"
        line = self.instance_to_line(text, index)

        instance = SentenceInstance.read_from_line(line)
        assert instance.text == text
        assert instance.index == index
        assert instance.label is None

    def test_end_to_end_conversion_to_arrays(self):
        instance = SentenceInstance("this is a sentence")
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad({'num_sentence_words': 7})
        word_array, label_array = indexed_instance.as_training_data()
        assert_array_equal(word_array, [0, 0, self.start_index, self.this_index, self.is_index,
                                        self.a_index, self.sentence_index])
        assert_array_equal(label_array, [[0], [0], [self.this_index], [self.is_index],
                                         [self.a_index], [self.sentence_index], [self.end_index]])

    def test_end_to_end_conversion_to_arrays_with_character_tokenizer(self):
        TextInstance.tokenizer = tokenizers['characters'](Params({}))
        instance = SentenceInstance("a sentence")
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad({'num_sentence_words': 13})
        word_array, label_array = indexed_instance.as_training_data()
        assert_array_equal(word_array, [0, 0, self.start_index, self.a_index, self.space_index,
                                        self.s_index, self.e_index, self.n_index, self.t_index,
                                        self.e_index, self.n_index, self.c_index, self.e_index])
        assert_array_equal(label_array, [[0], [0], [self.a_index], [self.space_index],
                                         [self.s_index], [self.e_index], [self.n_index],
                                         [self.t_index], [self.e_index], [self.n_index],
                                         [self.c_index], [self.e_index], [self.end_index]])

    def test_end_to_end_conversion_to_arrays_with_word_and_character_tokenizer(self):
        TextInstance.tokenizer = tokenizers['words and characters'](Params({}))
        instance = SentenceInstance("this is a sentence")
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad({'num_sentence_words': 6, 'num_word_characters': 5})
        word_array, label_array = indexed_instance.as_training_data()
        assert_array_equal(word_array, [[0, 0, 0, 0, 0],
                                        [self.start_index, 0, 0, 0, 0],
                                        [self.this_index, self.t_char_index, self.h_char_index,
                                         self.i_char_index, self.s_char_index],
                                        [self.is_index, self.i_char_index, self.s_char_index, 0, 0],
                                        [self.a_index, self.a_char_index, 0, 0, 0],
                                        [self.sentence_index, self.s_char_index, self.e_char_index,
                                         self.n_char_index, self.t_char_index],
                                       ])
        assert_array_equal(label_array, [[0], [self.this_index], [self.is_index], [self.a_index],
                                         [self.sentence_index], [self.end_index]])


class TestIndexedSentenceInstance(DeepQaTestCase):
    def test_get_padding_lengths_returns_length_of_word_indices(self):
        instance = IndexedSentenceInstance([1, 2, 3, 4], [2, 3, 4, 5])
        assert instance.get_padding_lengths() == {'num_sentence_words': 4}

    def test_pad_adds_zeros_on_left(self):
        instance = IndexedSentenceInstance([1, 2, 3, 4], [2, 3, 4, 5])
        instance.pad({'num_sentence_words': 5})
        assert instance.word_indices == [0, 1, 2, 3, 4]
        assert instance.label == [0, 2, 3, 4, 5]

    def test_pad_truncates_from_right(self):
        instance = IndexedSentenceInstance([1, 2, 3, 4], [2, 3, 4, 5])
        instance.pad({'num_sentence_words': 3})
        assert instance.word_indices == [2, 3, 4]
        assert instance.label == [3, 4, 5]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedSentenceInstance([1, 2, 3, 4], [2, 3, 4, 5])
        inputs, label = instance.as_training_data()
        assert_array_equal(inputs, [1, 2, 3, 4])
        assert_array_equal(label, [[2], [3], [4], [5]])

    def test_as_training_data_produces_correct_numpy_arrays_with_character_tokenization(self):
        instance = IndexedSentenceInstance([[1, 2], [3, 1, 2]], [3, 4])
        instance.pad({'num_sentence_words': 3, 'num_word_characters': 4})
        inputs, label = instance.as_training_data()
        assert_array_equal(inputs, [[0, 0, 0, 0], [1, 2, 0, 0], [3, 1, 2, 0]])
        assert_array_equal(label, [[0], [3], [4]])
