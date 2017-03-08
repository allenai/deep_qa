# pylint: disable=no-self-use,invalid-name
from typing import Tuple

import numpy

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.character_span_instance import CharacterSpanInstance
from ...common.test_case import DeepQaTestCase


class TestCharacterSpanInstance(DeepQaTestCase):
    @staticmethod
    def instance_to_line(question: str, passage: str, label: Tuple[int, int],
                         index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += (question + '\t' + passage + '\t' +
                 str(label[0]) + ',' + str(label[1]))
        return line

    def test_read_from_line_handles_three_column(self):
        question = "What do dogs eat?"
        passage = "Dogs eat cats."
        label = (9, 13)
        line = self.instance_to_line(question, passage, label)
        instance = CharacterSpanInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.passage_text == passage
        assert instance.label == label
        assert instance.index is None

    def test_read_from_line_handles_four_column(self):
        question = "What do dogs eat?"
        passage = "Dogs eat cats."
        label = (9, 13)
        index = 23
        line = self.instance_to_line(question, passage, label, index)
        instance = CharacterSpanInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.passage_text == passage
        assert instance.label == label
        assert instance.index == index

    def test_to_indexed_instance_converts_correctly(self):
        instance = CharacterSpanInstance("What do dogs eat?", "Dogs eat cats.",
                                         (9, 13))
        data_indexer = DataIndexer()
        what_index = data_indexer.add_word_to_index("what")
        do_index = data_indexer.add_word_to_index("do")
        dogs_index = data_indexer.add_word_to_index("dogs")
        eat_index = data_indexer.add_word_to_index("eat")
        cats_index = data_indexer.add_word_to_index("cats")
        period_index = data_indexer.add_word_to_index(".")
        question_index = data_indexer.add_word_to_index("?")
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.question_indices == [what_index, do_index,
                                                     dogs_index, eat_index,
                                                     question_index]
        assert indexed_instance.passage_indices == [dogs_index, eat_index,
                                                    cats_index, period_index]
        assert indexed_instance.label == (2, 2)

        # I put this test in here, instead of its own `test_as_training_data` test, to be sure that
        # the conversion to IndexedCharacterSpanIndex was performed correctly.
        indexed_instance.pad({'num_question_words': 3, 'num_passage_words': 6})
        (question_array, passage_array), label = indexed_instance.as_training_data()
        assert isinstance(label, tuple)
        assert numpy.all(label[0] == numpy.asarray([0, 0, 1, 0, 0, 0]))
        assert numpy.all(label[1] == numpy.asarray([0, 0, 1, 0, 0, 0]))
        assert numpy.all(question_array == numpy.asarray([dogs_index, eat_index, question_index]))
        assert numpy.all(passage_array == numpy.asarray([dogs_index, eat_index, cats_index,
                                                         period_index, 0, 0]))
