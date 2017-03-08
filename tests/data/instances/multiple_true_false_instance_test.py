# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.instances.background_instance import IndexedBackgroundInstance
from deep_qa.data.instances.multiple_true_false_instance import IndexedMultipleTrueFalseInstance
from deep_qa.data.instances.true_false_instance import IndexedTrueFalseInstance
from ...common.test_case import DeepQaTestCase


class TestIndexedMultipleTrueFalseInstance(DeepQaTestCase):
    def setUp(self):
        super(TestIndexedMultipleTrueFalseInstance, self).setUp()
        # We'll just test with underlying IndexedTrueFalseInstances for most of these, because it's
        # simpler.
        self.instance = IndexedMultipleTrueFalseInstance(
                [
                        IndexedTrueFalseInstance([1], False),
                        IndexedTrueFalseInstance([2, 3, 4], False),
                        IndexedTrueFalseInstance([5, 6], True),
                        IndexedTrueFalseInstance([7, 8], False)
                ],
                2)

    def test_get_lengths_returns_max_of_options(self):
        assert self.instance.get_lengths() == {'num_sentence_words': 3, 'num_options': 4}

    def test_pad_calls_pad_on_all_options(self):
        self.instance.pad({'num_sentence_words': 3, 'num_options': 4})
        assert self.instance.options[0].word_indices == [0, 0, 1]
        assert self.instance.options[1].word_indices == [2, 3, 4]
        assert self.instance.options[2].word_indices == [0, 5, 6]
        assert self.instance.options[3].word_indices == [0, 7, 8]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad({'num_sentence_words': 2, 'num_options': 5})
        assert self.instance.options[0].word_indices == [0, 1]
        assert self.instance.options[1].word_indices == [3, 4]
        assert self.instance.options[2].word_indices == [5, 6]
        assert self.instance.options[3].word_indices == [7, 8]
        assert self.instance.options[4].word_indices == [0, 0]
        assert len(self.instance.options) == 5

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad({'num_sentence_words': 1, 'num_options': 1})
        assert self.instance.options[0].word_indices == [1]
        assert len(self.instance.options) == 1

    def test_as_training_data_produces_correct_numpy_arrays_with_simple_instances(self):
        self.instance.pad({'num_sentence_words': 3, 'num_options': 4})
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 1, 0]))
        assert numpy.all(inputs == numpy.asarray([[0, 0, 1], [2, 3, 4], [0, 5, 6], [0, 7, 8]]))

    def test_as_training_data_produces_correct_numpy_arrays_with_background_instances(self):
        instance = IndexedMultipleTrueFalseInstance(
                [
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([1, 2], False),
                                                  [IndexedTrueFalseInstance([2], None),
                                                   IndexedTrueFalseInstance([3], None)]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([3, 4], False),
                                                  [IndexedTrueFalseInstance([5], None),
                                                   IndexedTrueFalseInstance([6], None)]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([5, 6], False),
                                                  [IndexedTrueFalseInstance([8], None),
                                                   IndexedTrueFalseInstance([9], None)]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([7, 8], True),
                                                  [IndexedTrueFalseInstance([11], None),
                                                   IndexedTrueFalseInstance([12], None)]),
                ],
                3)
        (word_arrays, background_arrays), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 0, 1]))
        assert numpy.all(word_arrays == numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]))
        assert numpy.all(background_arrays == numpy.asarray([[[2], [3]],
                                                             [[5], [6]],
                                                             [[8], [9]],
                                                             [[11], [12]]]))
