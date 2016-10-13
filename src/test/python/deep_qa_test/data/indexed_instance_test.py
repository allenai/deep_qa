# pylint: disable=no-self-use,invalid-name

from unittest import TestCase

import numpy

from deep_qa.data.indexed_instance import IndexedBackgroundInstance
from deep_qa.data.indexed_instance import IndexedMultipleChoiceInstance
from deep_qa.data.indexed_instance import IndexedQuestionAnswerInstance
from deep_qa.data.indexed_instance import IndexedSentencePairInstance
from deep_qa.data.indexed_instance import IndexedTrueFalseInstance


class TestIndexedTrueFalseInstance:
    def test_get_lengths_returns_length_of_word_indices(self):
        instance = IndexedTrueFalseInstance([1, 2, 3, 4], True)
        assert instance.get_lengths() == {'word_sequence_length': 4}

    def test_pad_adds_zeros_on_left(self):
        instance = IndexedTrueFalseInstance([1, 2, 3, 4], True)
        instance.pad({'word_sequence_length': 5})
        assert instance.word_indices == [0, 1, 2, 3, 4]

    def test_pad_truncates_from_right(self):
        instance = IndexedTrueFalseInstance([1, 2, 3, 4], True)
        instance.pad({'word_sequence_length': 3})
        assert instance.word_indices == [2, 3, 4]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedTrueFalseInstance([1, 2, 3, 4], True)
        inputs, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1]))
        assert numpy.all(inputs == numpy.asarray([1, 2, 3, 4]))
        instance.label = False
        _, label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([1, 0]))


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


class TestIndexedMultipleChoiceInstance(TestCase):
    def setUp(self):
        # We'll just test with underlying IndexedTrueFalseInstances for most of these, because it's
        # simpler.
        self.instance = IndexedMultipleChoiceInstance(
                [
                        IndexedTrueFalseInstance([1], False),
                        IndexedTrueFalseInstance([2, 3, 4], False),
                        IndexedTrueFalseInstance([5, 6], True),
                        IndexedTrueFalseInstance([7, 8], False)
                ],
                2)

    def test_get_lengths_returns_max_of_options(self):
        assert self.instance.get_lengths() == {'word_sequence_length': 3, 'num_options': 4}

    def test_pad_calls_pad_on_all_options(self):
        self.instance.pad({'word_sequence_length': 3, 'num_options': 4})
        assert self.instance.options[0].word_indices == [0, 0, 1]
        assert self.instance.options[1].word_indices == [2, 3, 4]
        assert self.instance.options[2].word_indices == [0, 5, 6]
        assert self.instance.options[3].word_indices == [0, 7, 8]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad({'word_sequence_length': 2, 'num_options': 5})
        assert self.instance.options[0].word_indices == [0, 1]
        assert self.instance.options[1].word_indices == [3, 4]
        assert self.instance.options[2].word_indices == [5, 6]
        assert self.instance.options[3].word_indices == [7, 8]
        assert self.instance.options[4].word_indices == [0, 0]
        assert len(self.instance.options) == 5

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad({'word_sequence_length': 1, 'num_options': 1})
        assert self.instance.options[0].word_indices == [1]
        assert len(self.instance.options) == 1

    def test_as_training_data_produces_correct_numpy_arrays_with_simple_instances(self):
        self.instance.pad({'word_sequence_length': 3, 'num_options': 4})
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 1, 0]))
        assert numpy.all(inputs == numpy.asarray([[0, 0, 1], [2, 3, 4], [0, 5, 6], [0, 7, 8]]))

    def test_as_training_data_produces_correct_numpy_arrays_with_background_instances(self):
        instance = IndexedMultipleChoiceInstance(
                [
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([1, 2], False),
                                                  [[2], [3]]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([3, 4], False),
                                                  [[5], [6]]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([5, 6], False),
                                                  [[8], [9]]),
                        IndexedBackgroundInstance(IndexedTrueFalseInstance([7, 8], True),
                                                  [[11], [12]]),
                ],
                3)
        (word_arrays, background_arrays), label = instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 0, 1]))
        assert numpy.all(word_arrays == numpy.asarray([[1, 2], [3, 4], [5, 6], [7, 8]]))
        assert numpy.all(background_arrays == numpy.asarray([[[2], [3]],
                                                             [[5], [6]],
                                                             [[8], [9]],
                                                             [[11], [12]]]))


class TestIndexedQuestionAnswerInstance(TestCase):
    def setUp(self):
        self.instance = IndexedQuestionAnswerInstance([1, 2, 3],
                                                      [[2, 3], [4], [5, 6]],
                                                      1)

    def test_get_lengths_returns_three_correct_lengths(self):
        assert self.instance.get_lengths() == {
                'word_sequence_length': 3,
                'answer_length': 2,
                'num_options': 3
                }

    def test_pad_calls_pad_on_all_options(self):
        self.instance.pad({'word_sequence_length': 2, 'answer_length': 2, 'num_options': 3})
        assert self.instance.question_indices == [2, 3]
        assert self.instance.option_indices[0] == [2, 3]
        assert self.instance.option_indices[1] == [0, 4]
        assert self.instance.option_indices[2] == [5, 6]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad({'word_sequence_length': 1, 'answer_length': 1, 'num_options': 4})
        assert self.instance.question_indices == [3]
        assert self.instance.option_indices[0] == [3]
        assert self.instance.option_indices[1] == [4]
        assert self.instance.option_indices[2] == [6]
        assert self.instance.option_indices[3] == [0]
        assert len(self.instance.option_indices) == 4

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad({'word_sequence_length': 1, 'answer_length': 1, 'num_options': 1})
        assert self.instance.question_indices == [3]
        assert self.instance.option_indices[0] == [3]
        assert len(self.instance.option_indices) == 1

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'word_sequence_length': 3, 'answer_length': 2, 'num_options': 3})
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1, 0]))
        assert numpy.all(inputs[0] == numpy.asarray([1, 2, 3]))
        assert numpy.all(inputs[1] == numpy.asarray([[2, 3], [0, 4], [5, 6]]))


class TestIndexedSentencePairInstance:
    def test_get_lengths_returns_max_of_text_and_hypothesis(self):
        instance = IndexedSentencePairInstance([1, 2, 3], [1], True)
        assert instance.get_lengths() == {'word_sequence_length': 3}
        instance = IndexedSentencePairInstance([1, 2, 3], [1, 2, 3, 4], True)
        assert instance.get_lengths() == {'word_sequence_length': 4}

    def test_pad_pads_both_text_and_hypothesis(self):
        instance = IndexedSentencePairInstance([1, 2], [3, 4], True)
        instance.pad({'word_sequence_length': 3})
        assert instance.first_sentence_indices == [0, 1, 2]
        assert instance.second_sentence_indices == [0, 3, 4]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        # pylint: disable=redefined-variable-type
        instance = IndexedSentencePairInstance([1, 2], [3, 4], [0, 1, 0])
        inputs, label = instance.as_training_data()
        assert isinstance(inputs, tuple)
        assert len(inputs) == 2
        assert numpy.all(inputs[0] == numpy.asarray([1, 2]))
        assert numpy.all(inputs[1] == numpy.asarray([3, 4]))
        assert numpy.all(label == numpy.asarray([0, 1, 0]))
