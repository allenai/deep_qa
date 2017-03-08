# pylint: disable=no-self-use,invalid-name
from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.instance import TextInstance
from deep_qa.data.tokenizers import tokenizers
from deep_qa.data.instances.true_false_instance import IndexedTrueFalseInstance, TrueFalseInstance

from ...common.test_case import DeepQaTestCase

class TestTextInstance(DeepQaTestCase):
    """
    The point of this test class is to test the TextEncoder used by the TextInstance, to be sure
    that we get what we expect when using character encoders, or word-and-character encoders.
    """
    def tearDown(self):
        super(TestTextInstance, self).tearDown()
        TextInstance.tokenizer = tokenizers['words']({})

    def test_words_tokenizes_the_sentence_correctly(self):
        t = TrueFalseInstance("This is a sentence.", None)
        assert t.words() == {'words': ['this', 'is', 'a', 'sentence', '.']}
        TextInstance.tokenizer = tokenizers['characters']({})
        assert t.words() == {'characters': ['T', 'h', 'i', 's', ' ', 'i', 's', ' ', 'a', ' ', 's',
                                            'e', 'n', 't', 'e', 'n', 'c', 'e', '.']}
        TextInstance.tokenizer = tokenizers['words and characters']({})
        assert t.words() == {'words': ['this', 'is', 'a', 'sentence', '.'],
                             'characters': ['t', 'h', 'i', 's', 'i', 's', 'a', 's', 'e', 'n', 't',
                                            'e', 'n', 'c', 'e', '.']}

    def test_to_indexed_instance_converts_correctly(self):
        data_indexer = DataIndexer()
        a_word_index = data_indexer.add_word_to_index("a", namespace='words')
        sentence_index = data_indexer.add_word_to_index("sentence", namespace='words')
        capital_a_index = data_indexer.add_word_to_index("A", namespace='characters')
        space_index = data_indexer.add_word_to_index(" ", namespace='characters')
        a_index = data_indexer.add_word_to_index("a", namespace='characters')
        s_index = data_indexer.add_word_to_index("s", namespace='characters')
        e_index = data_indexer.add_word_to_index("e", namespace='characters')
        n_index = data_indexer.add_word_to_index("n", namespace='characters')
        t_index = data_indexer.add_word_to_index("t", namespace='characters')
        c_index = data_indexer.add_word_to_index("c", namespace='characters')

        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [a_word_index, sentence_index]
        TextInstance.tokenizer = tokenizers['characters']({})
        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [capital_a_index, space_index, s_index, e_index, n_index, t_index,
                                         e_index, n_index, c_index, e_index]
        TextInstance.tokenizer = tokenizers['words and characters']({})
        instance = TrueFalseInstance("A sentence", None).to_indexed_instance(data_indexer)
        assert instance.word_indices == [[a_word_index, a_index],
                                         [sentence_index, s_index, e_index, n_index, t_index,
                                          e_index, n_index, c_index, e_index]]


class TestIndexedInstance(DeepQaTestCase):
    def test_get_lengths_works_with_words_and_characters(self):
        instance = IndexedTrueFalseInstance([[1, 2], [3, 1, 2]], True)
        assert instance.get_lengths() == {'num_sentence_words': 2, 'num_word_characters': 3}

    def test_pad_word_sequence_handles_words_and_characters(self):
        instance = IndexedTrueFalseInstance([[1, 2], [3, 1, 2]], True)
        padded = instance.pad_word_sequence(instance.word_indices,
                                            {'num_sentence_words': 3, 'num_word_characters': 4})
        assert padded == [[0, 0, 0, 0], [1, 2, 0, 0], [3, 1, 2, 0]]
