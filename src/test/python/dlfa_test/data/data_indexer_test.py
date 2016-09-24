# pylint: disable=no-self-use,invalid-name

from dlfa.data.data_indexer import DataIndexer
from dlfa.data.dataset import TextDataset
from dlfa.data.text_instance import TrueFalseInstance

class TestDataIndexer:
    def test_fit_word_dictionary_respects_min_count(self):
        instance = TrueFalseInstance("a a a a b b c c c", True)
        dataset = TextDataset([instance])
        data_indexer = DataIndexer()
        data_indexer.fit_word_dictionary(dataset, min_count=4)
        assert 'a' in data_indexer.words_in_index()
        assert 'b' not in data_indexer.words_in_index()
        assert 'c' not in data_indexer.words_in_index()

        data_indexer = DataIndexer()
        data_indexer.fit_word_dictionary(dataset, min_count=1)
        assert 'a' in data_indexer.words_in_index()
        assert 'b' in data_indexer.words_in_index()
        assert 'c' in data_indexer.words_in_index()

    def test_add_word_to_index_gives_consistent_results(self):
        data_indexer = DataIndexer()
        initial_vocab_size = data_indexer.get_vocab_size()
        word_index = data_indexer.add_word_to_index("word")
        assert "word" in data_indexer.words_in_index()
        assert data_indexer.get_word_index("word") == word_index
        assert data_indexer.get_word_from_index(word_index) == "word"
        assert data_indexer.get_vocab_size() == initial_vocab_size + 1

        # Now add it again, and make sure nothing changes.
        data_indexer.add_word_to_index("word")
        assert "word" in data_indexer.words_in_index()
        assert data_indexer.get_word_index("word") == word_index
        assert data_indexer.get_word_from_index(word_index) == "word"
        assert data_indexer.get_vocab_size() == initial_vocab_size + 1

    def test_unknown_token(self):
        # pylint: disable=protected-access
        # We're putting this behavior in a test so that the behavior is documented.  There is
        # solver code that depends in a small way on how we treat the unknown token, so any
        # breaking change to this behavior should break a test, so you know you've done something
        # that needs more consideration.
        data_indexer = DataIndexer()
        oov_token = data_indexer._oov_token
        oov_index = data_indexer.get_word_index(oov_token)
        assert oov_index == 1
        assert data_indexer.get_word_index("unseen word") == oov_index
