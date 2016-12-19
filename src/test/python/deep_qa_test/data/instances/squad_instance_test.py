# pylint: disable=no-self-use,invalid-name

from typing import Tuple

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.squad_instance import SquadInstance

class TestSquadInstance:
    @staticmethod
    def instance_to_line(question: str, passage: str, label: Tuple[int, int], index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += question + '\t' + passage + '\t' + str(label[0]) + ',' + str(label[1])
        return line

    def test_read_from_line_handles_three_column(self):
        question = "What do dogs eat?"
        passage = "Dogs eat cats."
        label = (9, 13)
        line = self.instance_to_line(question, passage, label)
        instance = SquadInstance.read_from_line(line)
        assert instance.first_sentence == question
        assert instance.second_sentence == passage
        assert instance.label == label
        assert instance.index is None

    def test_read_from_line_handles_four_column(self):
        question = "What do dogs eat?"
        passage = "Dogs eat cats."
        label = (9, 13)
        index = 23
        line = self.instance_to_line(question, passage, label, index)
        instance = SquadInstance.read_from_line(line)
        assert instance.first_sentence == question
        assert instance.second_sentence == passage
        assert instance.label == label
        assert instance.index == index

    def test_to_indexed_instance_converts_correctly(self):
        instance = SquadInstance("What do dogs eat?", "Dogs eat cats.", (9, 13))
        data_indexer = DataIndexer()
        what_index = data_indexer.add_word_to_index("what")
        do_index = data_indexer.add_word_to_index("do")
        dogs_index = data_indexer.add_word_to_index("dogs")
        eat_index = data_indexer.add_word_to_index("eat")
        cats_index = data_indexer.add_word_to_index("cats")
        period_index = data_indexer.add_word_to_index(".")
        question_index = data_indexer.add_word_to_index("?")
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.first_sentence_indices == [what_index, do_index, dogs_index,
                                                           eat_index, question_index]
        assert indexed_instance.second_sentence_indices == [dogs_index, eat_index, cats_index,
                                                            period_index]
        assert indexed_instance.label == (2, 2)
