# pylint: disable=no-self-use,invalid-name
import numpy as np

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.mc_question_answer_instance import McQuestionAnswerInstance
from deep_qa.data.instances.mc_question_answer_instance import IndexedMcQuestionAnswerInstance
from ...common.test_case import DeepQaTestCase


class TestMcQuestionAnswerInstance:
    @staticmethod
    def instance_to_line(passage: str, question: str,
                         options: str, label: int, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += passage + '\t' + question + '\t' + options + '\t'+ str(label)
        return line

    def test_words_has_question_passage_options(self):
        instance = McQuestionAnswerInstance("Cats from Nevada are eaten by dogs in XXX .",
                                            "Dogs eat cats from Nevada in Washington .",
                                            ["Nevada", "Washington"], 1)
        assert instance.words() == {'words': ['cats', 'from', 'nevada', 'are', 'eaten', 'by',
                                              'dogs', 'in', 'xxx', '.', 'dogs', 'eat', 'cats',
                                              'from', 'nevada', 'in', 'washington', '.', 'nevada',
                                              'washington']}

    def test_read_from_line_handles_five_column(self):
        passage = "Dogs eat cats from Nevada in Washington ."
        question = "Cats from Nevada are eaten by dogs in XXX ."
        options_str = "Nevada###Washington"
        label = 1
        line = self.instance_to_line(passage, question, options_str, label)
        instance = McQuestionAnswerInstance.read_from_line(line)

        assert instance.question_text == question
        assert instance.passage_text == passage
        options = ["Nevada", "Washington"]
        assert instance.answer_options == options
        assert instance.label == label
        assert instance.index is None

    def test_read_from_line_handles_six_column(self):
        passage = "Dogs eat cats from Nevada in Washington ."
        question = "Cats from Nevada are eaten by dogs in XXX ."
        options_str = "Nevada###Washington"
        label = 1
        index = 42
        line = self.instance_to_line(passage, question, options_str, label, index)
        instance = McQuestionAnswerInstance.read_from_line(line)

        assert instance.question_text == question
        assert instance.passage_text == passage
        options = ["Nevada", "Washington"]
        assert instance.answer_options == options
        assert instance.label == label
        assert instance.index == index

    def test_to_indexed_instance_converts_correctly(self):
        instance = McQuestionAnswerInstance("Cats from Nevada are eaten by dogs in XXX .",
                                            "Dogs eat cats from Nevada in Washington .",
                                            ["Nevada", "Washington"], 1)
        data_indexer = DataIndexer()
        cats_index = data_indexer.add_word_to_index("cats")
        are_index = data_indexer.add_word_to_index("are")
        eaten_index = data_indexer.add_word_to_index("eaten")
        by_index = data_indexer.add_word_to_index("by")
        dogs_index = data_indexer.add_word_to_index("dogs")
        in_index = data_indexer.add_word_to_index("in")
        XXX_index = data_indexer.add_word_to_index("xxx")
        period_index = data_indexer.add_word_to_index(".")
        eat_index = data_indexer.add_word_to_index("eat")
        from_index = data_indexer.add_word_to_index("from")
        nevada_index = data_indexer.add_word_to_index("nevada")
        washington_index = data_indexer.add_word_to_index("washington")
        indexed_instance = instance.to_indexed_instance(data_indexer)

        assert indexed_instance.question_indices == [cats_index, from_index,
                                                     nevada_index, are_index,
                                                     eaten_index, by_index,
                                                     dogs_index, in_index,
                                                     XXX_index, period_index]
        assert indexed_instance.passage_indices == [dogs_index, eat_index, cats_index,
                                                    from_index, nevada_index, in_index,
                                                    washington_index, period_index]
        assert len(indexed_instance.option_indices) == 2
        assert indexed_instance.option_indices[0] == [nevada_index]
        assert indexed_instance.option_indices[1] == [washington_index]
        assert indexed_instance.label == 1


class TestIndexedMcQuestionAnswerInstance(DeepQaTestCase):
    def setUp(self):
        super(TestIndexedMcQuestionAnswerInstance, self).setUp()
        self.instance = IndexedMcQuestionAnswerInstance([1, 2, 3, 5, 6],
                                                        [2, 3, 4, 5, 6, 7],
                                                        [[2], [3, 5], [6]],
                                                        1)

    def test_get_lengths_returns_three_correct_lengths(self):
        assert self.instance.get_lengths() == {
                'num_question_words': 5,
                'num_passage_words': 6,
                'num_option_words': 2,
                'num_options': 3
        }

    def test_pad_calls_pad_oun_all_options(self):
        self.instance.pad({'num_question_words': 7, 'num_passage_words': 9,
                           'num_option_words': 2, 'num_options': 3})
        assert self.instance.question_indices == [0, 0, 1, 2, 3, 5, 6]
        assert self.instance.passage_indices == [2, 3, 4, 5, 6, 7, 0, 0, 0]
        assert self.instance.option_indices[0] == [0, 2]
        assert self.instance.option_indices[1] == [3, 5]
        assert self.instance.option_indices[2] == [0, 6]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad({'num_question_words': 3, 'num_passage_words': 4,
                           'num_option_words': 1, 'num_options': 4})
        assert self.instance.question_indices == [3, 5, 6]
        assert self.instance.passage_indices == [2, 3, 4, 5]
        assert self.instance.option_indices[0] == [2]
        assert self.instance.option_indices[1] == [5]
        assert self.instance.option_indices[2] == [6]
        assert self.instance.option_indices[3] == [0]
        assert len(self.instance.option_indices) == 4

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad({'num_question_words': 3, 'num_passage_words': 4,
                           'num_option_words': 1, 'num_options': 1})
        assert self.instance.question_indices == [3, 5, 6]
        assert self.instance.passage_indices == [2, 3, 4, 5]
        assert self.instance.option_indices[0] == [2]
        assert len(self.instance.option_indices) == 1

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'num_question_words': 7, 'num_passage_words': 4,
                           'num_option_words': 2, 'num_options': 4})
        inputs, label = self.instance.as_training_data()
        assert np.all(label == np.asarray([0, 1, 0, 0]))
        assert np.all(inputs[0] == np.asarray([0, 0, 1, 2, 3, 5, 6]))
        assert np.all(inputs[1] == np.asarray([2, 3, 4, 5]))
        assert np.all(inputs[2] == np.asarray([[0, 2], [3, 5], [0, 6], [0, 0]]))
