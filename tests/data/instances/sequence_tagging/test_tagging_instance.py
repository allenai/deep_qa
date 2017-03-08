# pylint: disable=no-self-use,invalid-name
from numpy.testing import assert_array_almost_equal

from deep_qa.data.instances.sequence_tagging.tagging_instance import IndexedTaggingInstance
from ....common.test_case import DeepQaTestCase


class TestIndexedTaggingInstance(DeepQaTestCase):
    def setUp(self):
        super(TestIndexedTaggingInstance, self).setUp()
        self.instance = IndexedTaggingInstance([1, 2, 3, 4], [4, 5, 6])

    def test_get_lengths_returns_correct_lengths(self):
        assert self.instance.get_lengths() == {'num_sentence_words': 4}

    def test_pad_truncates_correctly(self):
        self.instance.pad({'num_sentence_words': 2})
        assert self.instance.text_indices == [1, 2]

    def test_pad_adds_padding_correctly(self):
        self.instance.pad({'num_sentence_words': 6})
        assert self.instance.text_indices == [1, 2, 3, 4, 0, 0]

    def test_as_training_data_produces_correct_arrays(self):
        text_array, label_array = self.instance.as_training_data()
        assert_array_almost_equal(text_array, [1, 2, 3, 4])
        assert_array_almost_equal(label_array, [4, 5, 6])
