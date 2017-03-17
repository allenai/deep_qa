# pylint: disable=no-self-use,invalid-name
from typing import List

import numpy

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.sentence_selection_instance import SentenceSelectionInstance
from deep_qa.data.instances.sentence_selection_instance import IndexedSentenceSelectionInstance
from ...common.test_case import DeepQaTestCase


class TestSentenceSelectionInstance(DeepQaTestCase):
    @staticmethod
    def instance_to_line(question: str, sentences: List[str],
                         label: int, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += (question + '\t' + '###'.join(sentences) + '\t' +
                 str(label))
        return line

    def test_read_from_line_handles_three_column(self):
        question = "What do dogs eat?"
        sentences = ["Dogs eat cats.", "Dogs play with cats.",
                     "Dogs enjoy cats."]
        label = 0
        line = self.instance_to_line(question, sentences, label)
        instance = SentenceSelectionInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.sentences == sentences
        assert instance.label == label
        assert instance.index is None

    def test_read_from_line_handles_four_column(self):
        question = "What do dogs eat?"
        sentences = ["Dogs eat cats.", "Dogs play with cats.",
                     "Dogs enjoy cats."]
        label = 0
        index = 23
        line = self.instance_to_line(question, sentences, label, index)
        instance = SentenceSelectionInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.sentences == sentences
        assert instance.label == label
        assert instance.index == index

    def test_to_indexed_instance_converts_correctly(self):
        instance = SentenceSelectionInstance("What do dogs eat?",
                                             ["Dogs eat cats.",
                                              "Dogs play with cats.",
                                              "Dogs enjoy cats."],
                                             0)
        data_indexer = DataIndexer()
        what_index = data_indexer.add_word_to_index("what")
        do_index = data_indexer.add_word_to_index("do")
        dogs_index = data_indexer.add_word_to_index("dogs")
        eat_index = data_indexer.add_word_to_index("eat")
        cats_index = data_indexer.add_word_to_index("cats")
        period_index = data_indexer.add_word_to_index(".")
        question_index = data_indexer.add_word_to_index("?")
        play_index = data_indexer.add_word_to_index("play")
        with_index = data_indexer.add_word_to_index("with")
        enjoy_index = data_indexer.add_word_to_index("enjoy")
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.question_indices == [what_index, do_index,
                                                     dogs_index, eat_index,
                                                     question_index]
        assert indexed_instance.sentences_indices == [[dogs_index, eat_index,
                                                       cats_index, period_index],
                                                      [dogs_index, play_index,
                                                       with_index, cats_index,
                                                       period_index],
                                                      [dogs_index, enjoy_index,
                                                       cats_index, period_index]]
        assert indexed_instance.label == 0


class TestIndexedSentenceSelectionInstance(DeepQaTestCase):
    def setUp(self):
        super(TestIndexedSentenceSelectionInstance, self).setUp()
        self.instance = IndexedSentenceSelectionInstance([1, 2, 3, 5, 6],
                                                         [[2, 3, 4],
                                                          [4, 6, 5, 1],
                                                          [1, 2, 3, 4, 5, 6]],
                                                         1)

    def test_get_lengths(self):
        assert self.instance.get_lengths() == {'num_question_words': 5,
                                               'num_sentences': 3,
                                               'num_sentence_words': 6}

    def test_pad_adds_empty_sentences(self):
        self.instance.pad({'num_question_words': 3,
                           'num_sentence_words': 2,
                           'num_sentences': 4})
        assert self.instance.question_indices == [3, 5, 6]
        assert self.instance.sentences_indices[0] == [3, 4]
        assert self.instance.sentences_indices[1] == [5, 1]
        assert self.instance.sentences_indices[2] == [5, 6]
        assert self.instance.sentences_indices[3] == [0, 0]

    def test_pad_removes_sentences(self):
        self.instance.pad({'num_question_words': 4,
                           'num_sentence_words': 3,
                           'num_sentences': 2})
        assert self.instance.question_indices == [2, 3, 5, 6]
        assert self.instance.sentences_indices[0] == [2, 3, 4]
        assert self.instance.sentences_indices[1] == [6, 5, 1]

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'num_question_words': 7,
                           'num_sentence_words': 4,
                           'num_sentences': 4})
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1, 0, 0]))
        assert numpy.all(inputs[0] == numpy.asarray([0, 0, 1, 2, 3, 5, 6]))
        assert numpy.all(inputs[1] == numpy.asarray([[0, 2, 3, 4],
                                                     [4, 6, 5, 1],
                                                     [3, 4, 5, 6],
                                                     [0, 0, 0, 0]]))
