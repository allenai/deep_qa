# pylint: disable=no-self-use,invalid-name
from typing import List

from numpy.testing import assert_array_almost_equal

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.instance import TextInstance
from deep_qa.data.instances.sequence_tagging.pretokenized_tagging_instance import PreTokenizedTaggingInstance
from deep_qa.data.tokenizers import tokenizers
from ....common.test_case import DeepQaTestCase


class TestPreTokenizedTaggingInstance(DeepQaTestCase):
    def setUp(self):
        super(TestPreTokenizedTaggingInstance, self).setUp()
        tokens = ["cats", "are", "animals", "."]
        tags = ["N", "V", "N", "."]
        self.instance = PreTokenizedTaggingInstance(tokens, tags)
        TextInstance.tokenizer = tokenizers['words']({'processor': {'word_splitter': 'no_op'}})

    def tearDown(self):
        super(TestPreTokenizedTaggingInstance, self).tearDown()
        TextInstance.tokenizer = tokenizers['words']({})

    @staticmethod
    def instance_to_line(tokens: List[str], tags: List[str], index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        tagged_tokens = [token + '###' + tag for token, tag in zip(tokens, tags)]
        line += '\t'.join(tagged_tokens)
        return line

    def test_read_from_line_handles_example_indices(self):
        tokens = ["cats", "are", "animals", "."]
        tags = ["N", "V", "N", "."]
        index = 4
        line = self.instance_to_line(tokens, tags, index)
        instance = PreTokenizedTaggingInstance.read_from_line(line)
        assert instance.text == tokens
        assert instance.label == tags
        assert instance.index == index

    def test_read_from_line_handles_no_indices(self):
        tokens = ["cats", "are", "animals", "."]
        tags = ["N", "V", "N", "."]
        index = None
        line = self.instance_to_line(tokens, tags, index)
        instance = PreTokenizedTaggingInstance.read_from_line(line)
        assert instance.text == tokens
        assert instance.label == tags
        assert instance.index == index

    def test_to_indexed_instance_converts_correctly(self):
        data_indexer = DataIndexer()
        cats_index = data_indexer.add_word_to_index("cats")
        are_index = data_indexer.add_word_to_index("are")
        animals_index = data_indexer.add_word_to_index("animals")
        period_index = data_indexer.add_word_to_index(".")
        n_tag_index = data_indexer.add_word_to_index("N", namespace="tags")
        v_tag_index = data_indexer.add_word_to_index("V", namespace="tags")
        period_tag_index = data_indexer.add_word_to_index(".", namespace="tags")
        indexed_instance = self.instance.to_indexed_instance(data_indexer)
        assert indexed_instance.text_indices == [cats_index, are_index, animals_index, period_index]
        expected_label = [self.one_hot(n_tag_index - 2, 3),
                          self.one_hot(v_tag_index - 2, 3),
                          self.one_hot(n_tag_index - 2, 3),
                          self.one_hot(period_tag_index - 2, 3)]
        assert_array_almost_equal(indexed_instance.label, expected_label)

    def test_words_returns_correct_dictionary(self):
        assert self.instance.words() == {'words': ['cats', 'are', 'animals', '.'],
                                         'tags': ['N', 'V', 'N', '.']}
