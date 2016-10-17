# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

import numpy

from deep_qa.data.instances.true_false_instance import IndexedTrueFalseInstance
from deep_qa.data.instances.question_answer_instance import IndexedQuestionAnswerInstance
from deep_qa.data.instances.background_instance import IndexedBackgroundInstance

class TestIndexedBackgroundInstance(TestCase):
    def setUp(self):
        self.base_instance = IndexedTrueFalseInstance([1, 2], True)
        self.qa_instance = IndexedQuestionAnswerInstance([1, 2, 3],
                                                         [[2, 3], [4], [5, 6]],
                                                         1)

    def test_get_lengths_returns_max_of_background_and_word_indices(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2, 3, 4], [4, 5]])
        assert instance.get_lengths()['word_sequence_length'] == 3

    def test_get_lengths_returns_correct_background_length(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2, 3, 4], [4, 5]])
        assert instance.get_lengths() == {'word_sequence_length': 3, 'background_sentences': 2}

    def test_pad_adds_zeros_on_left_to_background(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2, 3]])
        instance.pad({'word_sequence_length': 3, 'background_sentences': 1})
        assert instance.indexed_instance.word_indices == [0, 1, 2]
        assert instance.background_indices == [[0, 2, 3]]

    def test_pad_truncates_from_right_on_background(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2, 3]])
        instance.pad({'word_sequence_length': 1, 'background_sentences': 1})
        assert instance.indexed_instance.word_indices == [2]
        assert instance.background_indices == [[3]]

    def test_pad_adds_padded_background_at_end(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2]])
        instance.pad({'word_sequence_length': 2, 'background_sentences': 2})
        assert instance.indexed_instance.word_indices == [1, 2]
        assert instance.background_indices == [[0, 2], [0, 0]]

    def test_pad_truncates_background_from_left(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2], [3]])
        instance.pad({'word_sequence_length': 1, 'background_sentences': 1})
        assert instance.indexed_instance.word_indices == [2]
        assert instance.background_indices == [[2]]

    def test_pad_works_with_complex_contained_instance(self):
        instance = IndexedBackgroundInstance(self.qa_instance, [[2]])
        instance.pad({
                'word_sequence_length': 3,
                'answer_length': 1,
                'num_options': 2,
                'background_sentences': 2,
                })
        assert instance.indexed_instance.question_indices == [1, 2, 3]
        assert len(instance.indexed_instance.option_indices) == 2
        assert instance.indexed_instance.option_indices[0] == [3]
        assert instance.indexed_instance.option_indices[1] == [4]
        assert instance.background_indices == [[0, 0, 2], [0, 0, 0]]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedBackgroundInstance(self.base_instance, [[2, 3], [4, 5]])
        (word_array, background_array), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1]))
        assert numpy.all(word_array == numpy.asarray([1, 2]))
        assert numpy.all(background_array == numpy.asarray([[2, 3], [4, 5]]))
        instance.indexed_instance.label = False
        _, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([1, 0]))
        instance.label = True  # we ignore this label, only using the indexed_instance label
        _, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([1, 0]))

    def test_as_training_data_produces_correct_numpy_arrays_with_complex_contained_instance(self):
        # We need the background array to always be _second_, not last.
        instance = IndexedBackgroundInstance(self.qa_instance, [[2, 3], [4, 5]])
        instance.pad({
                'word_sequence_length': 2,
                'answer_length': 2,
                'num_options': 3,
                'background_sentences': 2,
                })
        (question_array, background_array, answer_array), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1, 0]))
        assert numpy.all(question_array == numpy.asarray([2, 3]))
        assert numpy.all(answer_array == numpy.asarray([[2, 3], [0, 4], [5, 6]]))
        assert numpy.all(background_array == numpy.asarray([[2, 3], [4, 5]]))
