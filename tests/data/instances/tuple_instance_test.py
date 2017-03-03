# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.instances.tuple_instance import IndexedTupleInstance, TupleInstance
from ...common.test_case import DeepQaTestCase


class TestTupleInstance:
    def test_read_from_line_splits_correctly(self):
        line = "this###is###a triple"
        instance = TupleInstance.read_from_line(line)
        assert instance.text == ["this", "is", "a triple"]

    def test_read_from_line_handles_multiple_columns(self):
        index = 23
        tuple_string = "this###is###a triple"
        label = True
        line = "%d\t%s\t%d" % (index, tuple_string, int(label))
        instance = TupleInstance.read_from_line(line)
        assert instance.text == ["this", "is", "a triple"]
        assert instance.label is label
        assert instance.index == index

    def test_words_tokenizes_the_tuple_correctly(self):
        t = TupleInstance(["This", "is", "a triple"])
        assert t.words() == {'words': ['this', 'is', 'a', 'triple']}


class TestIndexedTupleInstance(DeepQaTestCase):
    def test_get_lengths_returns_length_of_longest_slot(self):
        instance = IndexedTupleInstance([[1, 2], [3, 4, 5], [6]], True)
        assert instance.get_lengths() == {'word_sequence_length': 3, 'num_slots': 3}

    def test_pad_adds_zeros_on_all_slots(self):
        instance = IndexedTupleInstance([[1, 2], [3, 4, 5], [6]], True)
        instance.pad({'word_sequence_length': 4, 'num_slots': 3})
        assert instance.word_indices == [[0, 0, 1, 2], [0, 3, 4, 5], [0, 0, 0, 6]]

    def test_pad_slots_concatenates_at_end(self):
        instance = IndexedTupleInstance([[1, 2], [3, 4], [5, 6], [7, 8]], True)
        instance.pad({'word_sequence_length': 4, 'num_slots': 3})
        assert instance.word_indices == [[0, 0, 1, 2], [0, 0, 3, 4], [5, 6, 7, 8]]

    def test_pad_adjusts_slots_before_length(self):
        instance = IndexedTupleInstance([[1, 2], [3, 4], [5, 6], [7, 8]], True)
        instance.pad({'word_sequence_length': 2, 'num_slots': 3})
        print(instance.word_indices)
        assert instance.word_indices == [[1, 2], [3, 4], [7, 8]]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        instance = IndexedTupleInstance([[1, 2], [3, 4], [5, 6]], True)
        inputs, _ = instance.as_training_data()
        assert numpy.all(inputs == numpy.asarray([[1, 2], [3, 4], [5, 6]]))
