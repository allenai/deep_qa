# pylint: disable=no-self-use,invalid-name
from typing import List

import numpy

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.question_answer_instance import IndexedQuestionAnswerInstance
from deep_qa.data.instances.question_answer_instance import QuestionAnswerInstance
from ...common.test_case import DeepQaTestCase


class TestQuestionAnswerInstance(DeepQaTestCase):
    def instance_to_line(self, question: str, answers: List[str], label: int, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += question
        line += '\t'
        line += '###'.join(answers)
        line += '\t'
        line += str(label)
        return line

    def test_read_from_line_handles_three_column(self):
        question = "what is the answer"
        answers = ['a', 'b', 'c']
        label = 1
        line = self.instance_to_line(question, answers, label)
        instance = QuestionAnswerInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.answer_options == answers
        assert instance.label is label
        assert instance.index is None

    def test_read_from_line_handles_four_column(self):
        question = "what is the answer"
        answers = ['a', 'b', 'c']
        label = 1
        index = 23
        line = self.instance_to_line(question, answers, label, index)
        instance = QuestionAnswerInstance.read_from_line(line)
        assert instance.question_text == question
        assert instance.answer_options == answers
        assert instance.label is label
        assert instance.index is index

    def test_words_includes_question_and_answers(self):
        instance = QuestionAnswerInstance("a b c", ["d", "e f"], 1)
        assert instance.words() == {'words': ['a', 'b', 'c', 'd', 'e', 'f']}

    def test_to_indexed_instance_converts_correctly(self):
        instance = QuestionAnswerInstance("a A b", ["d", "e f D"], 1)
        data_indexer = DataIndexer()
        a_index = data_indexer.add_word_to_index("a")
        d_index = data_indexer.add_word_to_index("d")
        oov_index = data_indexer.get_word_index(data_indexer._oov_token)  # pylint: disable=protected-access
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.question_indices == [a_index, a_index, oov_index]
        assert len(indexed_instance.option_indices) == 2
        assert indexed_instance.option_indices[0] == [d_index]
        assert indexed_instance.option_indices[1] == [oov_index, oov_index, d_index]


class TestIndexedQuestionAnswerInstance(DeepQaTestCase):
    def setUp(self):
        super(TestIndexedQuestionAnswerInstance, self).setUp()
        self.instance = IndexedQuestionAnswerInstance([1, 2, 3],
                                                      [[2, 3], [4], [5, 6]],
                                                      1)

    def test_get_lengths_returns_three_correct_lengths(self):
        assert self.instance.get_lengths() == {
                'num_sentence_words': 3,
                'answer_length': 2,
                'num_options': 3
                }

    def test_pad_calls_pad_on_all_options(self):
        self.instance.pad({'num_sentence_words': 2, 'answer_length': 2, 'num_options': 3})
        assert self.instance.question_indices == [2, 3]
        assert self.instance.option_indices[0] == [2, 3]
        assert self.instance.option_indices[1] == [0, 4]
        assert self.instance.option_indices[2] == [5, 6]

    def test_pad_adds_empty_options_when_necessary(self):
        self.instance.pad({'num_sentence_words': 1, 'answer_length': 1, 'num_options': 4})
        assert self.instance.question_indices == [3]
        assert self.instance.option_indices[0] == [3]
        assert self.instance.option_indices[1] == [4]
        assert self.instance.option_indices[2] == [6]
        assert self.instance.option_indices[3] == [0]
        assert len(self.instance.option_indices) == 4

    def test_pad_removes_options_when_necessary(self):
        self.instance.pad({'num_sentence_words': 1, 'answer_length': 1, 'num_options': 1})
        assert self.instance.question_indices == [3]
        assert self.instance.option_indices[0] == [3]
        assert len(self.instance.option_indices) == 1

    def test_as_training_data_produces_correct_numpy_arrays(self):
        self.instance.pad({'num_sentence_words': 3, 'answer_length': 2, 'num_options': 3})
        inputs, label = self.instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 1, 0]))
        assert numpy.all(inputs[0] == numpy.asarray([1, 2, 3]))
        assert numpy.all(inputs[1] == numpy.asarray([[2, 3], [0, 4], [5, 6]]))
