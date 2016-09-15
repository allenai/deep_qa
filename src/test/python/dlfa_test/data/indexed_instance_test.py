# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

import numpy

from dlfa.data.instance import IndexedInstance, IndexedBackgroundInstance, IndexedQuestionInstance

class TestIndexedInstance:
    def test_get_lengths_returns_length_of_word_indices(self):
        instance = IndexedInstance([1, 2, 3, 4], True)
        assert instance.get_lengths() == [4]

    def test_pad_adds_zeros_on_left(self):
        instance = IndexedInstance([1, 2, 3, 4], True)
        instance.pad([5])
        assert instance.word_indices == [0, 1, 2, 3, 4]

    def test_pad_truncates_from_right(self):
        instance = IndexedInstance([1, 2, 3, 4], True)
        instance.pad([3])
        assert instance.word_indices == [2, 3, 4]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedInstance([1, 2, 3, 4], True)
        inputs, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1]))
        assert numpy.all(inputs == numpy.asarray([1, 2, 3, 4]))
        instance.label = False
        _, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([1, 0]))


class TestIndexedBackgroundInstance:
    def test_get_lengths_returns_max_of_background_and_word_indices(self):
        instance = IndexedBackgroundInstance([1, 2], [[2, 3, 4], [4, 5]], True)
        assert instance.get_lengths()[0] == 3

    def test_get_lengths_returns_correct_background_length(self):
        instance = IndexedBackgroundInstance([1, 2], [[2, 3, 4], [4, 5]], True)
        assert instance.get_lengths() == [3, 2]

    def test_pad_adds_zeros_on_left_to_background(self):
        instance = IndexedBackgroundInstance([1, 2], [[2, 3]], True)
        instance.pad([3, 1])
        assert instance.word_indices == [0, 1, 2]
        assert instance.background_indices == [[0, 2, 3]]

    def test_pad_truncates_from_right_on_background(self):
        instance = IndexedBackgroundInstance([1, 2], [[2, 3]], True)
        instance.pad([1, 1])
        assert instance.word_indices == [2]
        assert instance.background_indices == [[3]]

    def test_pad_adds_padded_background_at_end(self):
        instance = IndexedBackgroundInstance([1, 2], [[2]], True)
        instance.pad([2, 2])
        assert instance.word_indices == [1, 2]
        assert instance.background_indices == [[0, 2], [0, 0]]

    def test_pad_truncates_background_from_left(self):
        instance = IndexedBackgroundInstance([1], [[2], [3]], True)
        instance.pad([1, 1])
        assert instance.word_indices == [1]
        assert instance.background_indices == [[2]]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedBackgroundInstance([1, 2], [[2, 3], [4, 5]], True)
        (word_array, background_array), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1]))
        assert numpy.all(word_array == numpy.asarray([1, 2]))
        assert numpy.all(background_array == numpy.asarray([[2, 3], [4, 5]]))
        instance.label = False
        _, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([1, 0]))


class TestIndexedQuestionInstance(TestCase):

    def setUp(self):
        # We'll just test with underlying IndexedInstances for most of these, because it's simpler
        self.instance = IndexedQuestionInstance(
                [
                        IndexedInstance([1], False),
                        IndexedInstance([2, 3, 4], False),
                        IndexedInstance([5, 6], True),
                        IndexedInstance([7, 8], False)
                ],
                2)

    def test_get_lengths_returns_max_of_options(self):
        assert self.instance.get_lengths() == [3, 4]

    def test_pad_calls_pad_on_all_options(self):
        self.instance.pad([3, 4])
        assert self.instance.options[0].word_indices == [0, 0, 1]
        assert self.instance.options[1].word_indices == [2, 3, 4]
        assert self.instance.options[2].word_indices == [0, 5, 6]
        assert self.instance.options[3].word_indices == [0, 7, 8]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad([2, 5])
        assert self.instance.options[0].word_indices == [0, 1]
        assert self.instance.options[1].word_indices == [3, 4]
        assert self.instance.options[2].word_indices == [5, 6]
        assert self.instance.options[3].word_indices == [7, 8]
        assert self.instance.options[4].word_indices == [0, 0]
        assert len(self.instance.options) == 5

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad([1, 1])
        assert self.instance.options[0].word_indices == [1]
        assert len(self.instance.options) == 1

    def test_as_training_data_produces_correct_numpy_arrays_with_simple_instances(self):
        self.instance.pad([3, 4])
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 1, 0]))
        assert numpy.all(inputs == numpy.asarray([[0, 0, 1], [2, 3, 4], [0, 5, 6], [0, 7, 8]]))

    def test_as_training_data_produces_correct_numpy_arrays_with_background_instances(self):
        instance = IndexedQuestionInstance(
                [
                        IndexedBackgroundInstance([1, 2], [[2], [3]], False),
                        IndexedBackgroundInstance([3, 4], [[5], [6]], False),
                        IndexedBackgroundInstance([5, 6], [[8], [9]], False),
                        IndexedBackgroundInstance([7, 8], [[11], [12]], True),
                ],
                3)
        (word_arrays, background_arrays), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 0, 1]))
        assert numpy.all(word_arrays == numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]))
        assert numpy.all(background_arrays == numpy.asarray([[[2], [3]],
                                                             [[5], [6]],
                                                             [[8], [9]],
                                                             [[11], [12]]]))
