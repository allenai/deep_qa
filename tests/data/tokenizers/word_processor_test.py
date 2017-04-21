# pylint: disable=no-self-use,invalid-name

from deep_qa.data.tokenizers.word_processor import WordProcessor
from deep_qa.common.params import Params

class TestWordProcessor:
    def test_passes_through_correctly(self):
        word_processor = WordProcessor(Params({}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        tokens = word_processor.get_tokens(sentence)
        expected_tokens = ["this", "(", "sentence", ")", "has", "'", "crazy", "'", "\"",
                           "punctuation", "\"", "."]
        assert tokens == expected_tokens

    def test_stems_and_filters_correctly(self):
        word_processor = WordProcessor(Params({'word_stemmer': 'porter', 'word_filter': 'stopwords'}))
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["sentenc", "ha", "crazi", "punctuat"]
        tokens = word_processor.get_tokens(sentence)
        assert tokens == expected_tokens
