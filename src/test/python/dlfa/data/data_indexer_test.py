# pylint: disable=no-self-use,invalid-name

from dlfa.data.data_indexer import DataIndexer
from dlfa.data.dataset import TextDataset
from dlfa.data.instance import TextInstance

class TestDataIndexer:
    def test_fit_word_dictionary_respects_min_count(self):
        instance = TextInstance("a a a a b b c c c", True)
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
