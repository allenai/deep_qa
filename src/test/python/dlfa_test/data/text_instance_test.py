# pylint: disable=no-self-use,invalid-name

from typing import List

from dlfa.data.data_indexer import DataIndexer
from dlfa.data.text_instance import QuestionAnswerInstance, TrueFalseInstance

class TestTrueFalseInstance:
    @staticmethod
    def instance_to_line(text, label=None, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += text
        if label is not None:
            label_str = '1' if label else '0'
            line += '\t' + label_str
        return line

    def test_read_from_line_handles_one_column(self):
        text = "this is a sentence"
        instance = TrueFalseInstance.read_from_line(text)
        assert instance.text == text
        assert instance.label is None
        assert instance.index is None

    def test_read_from_line_handles_three_column(self):
        index = 23
        text = "this is a sentence"
        label = True
        line = self.instance_to_line(text, label, index)

        instance = TrueFalseInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_label(self):
        index = None
        text = "this is a sentence"
        label = True
        line = self.instance_to_line(text, label, index)

        instance = TrueFalseInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_index(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TrueFalseInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_default_true(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TrueFalseInstance.read_from_line(line, default_label=True)
        assert instance.text == text
        assert instance.label is True
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_default_false(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TrueFalseInstance.read_from_line(line, default_label=False)
        assert instance.text == text
        assert instance.label is False
        assert instance.index == index

    def test_words_tokenizes_the_sentence_correctly(self):
        t = TrueFalseInstance("This is a sentence.", None)
        assert t.words() == ['this', 'is', 'a', 'sentence', '.']
        t = TrueFalseInstance("This isn't a sentence.", None)
        assert t.words() == ['this', 'is', "n't", 'a', 'sentence', '.']
        t = TrueFalseInstance("And, I have commas.", None)
        assert t.words() == ['and', ',', 'i', 'have', 'commas', '.']


class TestQuestionAnswerInstance:
    @staticmethod
    def instance_to_line(question: str, answers: List[str], label: int, index=None):
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
        assert instance.words() == ['a', 'b', 'c', 'd', 'e', 'f']

    def test_to_indexed_instance_converts_correctly(self):
        instance = QuestionAnswerInstance("a b", ["d", "e f"], 1)
        data_indexer = DataIndexer()
        a_index = data_indexer.add_word_to_index("a")
        d_index = data_indexer.add_word_to_index("d")
        oov_index = data_indexer.get_word_index(data_indexer._oov_token)  # pylint: disable=protected-access
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.question_indices == [a_index, oov_index]
        assert len(indexed_instance.option_indices) == 2
        assert indexed_instance.option_indices[0] == [d_index]
        assert indexed_instance.option_indices[1] == [oov_index, oov_index]
