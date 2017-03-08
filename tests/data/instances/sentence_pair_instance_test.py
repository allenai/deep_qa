# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.instances.sentence_pair_instance import IndexedSentencePairInstance
from ...common.test_case import DeepQaTestCase


class TestIndexedSentencePairInstance(DeepQaTestCase):
    def test_get_lengths_returns_max_of_both_sentences(self):
        instance = IndexedSentencePairInstance([1, 2, 3], [1], True)
        assert instance.get_lengths() == {'num_sentence_words': 3}
        instance = IndexedSentencePairInstance([1, 2, 3], [1, 2, 3, 4], True)
        assert instance.get_lengths() == {'num_sentence_words': 4}

    def test_pad_pads_both_sentences(self):
        instance = IndexedSentencePairInstance([1, 2], [3, 4], True)
        instance.pad({'num_sentence_words': 3})
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
