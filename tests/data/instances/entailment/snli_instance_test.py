# pylint: disable=no-self-use,invalid-name

import pytest

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.entailment.snli_instance import SnliInstance

class TestSnliInstance:
    @staticmethod
    def instance_to_line(text: str, hypothesis: str, label: str, index=None):
        line = ''
        if index is not None:
            line += str(index) + '\t'
        line += text + '\t' + hypothesis + '\t' + label
        return line

    def test_read_from_line_handles_three_column(self):
        text = "dogs eat cats"
        hypothesis = "animals eat animals"
        label = "contradicts"
        line = self.instance_to_line(text, hypothesis, label)
        instance = SnliInstance.read_from_line(line)
        assert instance.first_sentence == text
        assert instance.second_sentence == hypothesis
        assert instance.label == [0, 1, 0]
        assert instance.index is None

    def test_read_from_line_handles_four_column(self):
        text = "dogs eat cats"
        hypothesis = "animals eat animals"
        label = "entails"
        index = 23
        line = self.instance_to_line(text, hypothesis, label, index)
        instance = SnliInstance.read_from_line(line)
        assert instance.first_sentence == text
        assert instance.second_sentence == hypothesis
        assert instance.label == [1, 0, 0]
        assert instance.index == index

    def test_words_includes_text_and_hypothesis(self):
        instance = SnliInstance("a b c", "d a", "entails")
        assert instance.words() == {'words': ['a', 'b', 'c', 'd', 'a']}

    def test_labels_are_mapped_correctly(self):
        assert SnliInstance("", "", "entails").label == [1, 0, 0]
        assert SnliInstance("", "", "contradicts").label == [0, 1, 0]
        assert SnliInstance("", "", "neutral").label == [0, 0, 1]
        assert SnliInstance("", "", "entails_softmax").label == [0, 1]
        assert SnliInstance("", "", "not_entails_softmax").label == [1, 0]
        assert SnliInstance("", "", "entails_sigmoid").label == [1]
        assert SnliInstance("", "", "not_entails_sigmoid").label == [0]
        assert SnliInstance("", "", "attention_true").label == [1]
        assert SnliInstance("", "", "attention_false").label == [0]

    def test_to_attention_instance_maps_label_correctly(self):
        assert SnliInstance("", "", "entails").to_attention_instance().label == [1]
        assert SnliInstance("", "", "contradicts").to_attention_instance().label == [1]
        assert SnliInstance("", "", "neutral").to_attention_instance().label == [0]
        with pytest.raises(Exception):
            SnliInstance("", "", True).to_attention_instance()
        with pytest.raises(Exception):
            SnliInstance("", "", False).to_attention_instance()

    def test_to_entails_instance_maps_label_correctly(self):
        assert SnliInstance("", "", "entails").to_entails_instance("softmax").label == [0, 1]
        assert SnliInstance("", "", "contradicts").to_entails_instance("softmax").label == [1, 0]
        assert SnliInstance("", "", "neutral").to_entails_instance("softmax").label == [1, 0]
        for label in SnliInstance.label_mapping:
            if label not in ["entails", "contradicts", "neutral"]:
                with pytest.raises(Exception):
                    SnliInstance("", "", label).to_entails_instance("softmax")
        assert SnliInstance("", "", "entails").to_entails_instance("sigmoid").label == [1]
        assert SnliInstance("", "", "contradicts").to_entails_instance("sigmoid").label == [0]
        assert SnliInstance("", "", "neutral").to_entails_instance("sigmoid").label == [0]
        for label in SnliInstance.label_mapping:
            if label not in ["entails", "contradicts", "neutral"]:
                with pytest.raises(Exception):
                    SnliInstance("", "", label).to_entails_instance("sigmoid")

    def test_to_indexed_instance_converts_correctly(self):
        instance = SnliInstance("a b", "d e f", "entails")
        data_indexer = DataIndexer()
        a_index = data_indexer.add_word_to_index("a")
        d_index = data_indexer.add_word_to_index("d")
        oov_index = data_indexer.get_word_index(data_indexer._oov_token)  # pylint: disable=protected-access
        indexed_instance = instance.to_indexed_instance(data_indexer)
        assert indexed_instance.first_sentence_indices == [a_index, oov_index]
        assert indexed_instance.second_sentence_indices == [d_index, oov_index, oov_index]
        assert indexed_instance.label == instance.label
