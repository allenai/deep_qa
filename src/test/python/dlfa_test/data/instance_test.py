# pylint: disable=no-self-use,invalid-name

from dlfa.data.instance import TextInstance

class TestTextInstance:
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
        instance = TextInstance.read_from_line(text)
        assert instance.text == text
        assert instance.label is None
        assert instance.index is None

    def test_read_from_line_handles_three_column(self):
        index = 23
        text = "this is a sentence"
        label = True
        line = self.instance_to_line(text, label, index)

        instance = TextInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_label(self):
        index = None
        text = "this is a sentence"
        label = True
        line = self.instance_to_line(text, label, index)

        instance = TextInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_index(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TextInstance.read_from_line(line)
        assert instance.text == text
        assert instance.label is label
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_default_true(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TextInstance.read_from_line(line, default_label=True)
        assert instance.text == text
        assert instance.label is True
        assert instance.index == index

    def test_read_from_line_handles_two_column_with_default_false(self):
        index = 23
        text = "this is a sentence"
        label = None
        line = self.instance_to_line(text, label, index)

        instance = TextInstance.read_from_line(line, default_label=False)
        assert instance.text == text
        assert instance.label is False
        assert instance.index == index

    def test_words_tokenizes_the_sentence_correctly(self):
        t = TextInstance("This is a sentence.", None)
        assert t.words() == ['this', 'is', 'a', 'sentence', '.']
        t = TextInstance("This isn't a sentence.", None)
        assert t.words() == ['this', 'is', "n't", 'a', 'sentence', '.']
        t = TextInstance("And, I have commas.", None)
        assert t.words() == ['and', ',', 'i', 'have', 'commas', '.']
