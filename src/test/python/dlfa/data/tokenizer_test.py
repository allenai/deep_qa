# pylint: disable=no-self-use,invalid-name

from dlfa.data.tokenizer import Tokenizer


class TestTokenizer:
    def test_tokenize_handles_complex_punctuation(self):
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["this", "(", "sentence", ")", "has", "'", "crazy", "'", '"',
                           "punctuation", '"', "."]
        tokens = Tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_contraction(self):
        sentence = "it ain't joe's problem; would've been yesterday"
        expected_tokens = ["it", "ai", "n't", "joe", "'s", "problem", ";", "would", "'ve", "been",
                           "yesterday"]
        tokens = Tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_multiple_contraction(self):
        sentence = "wouldn't've"
        expected_tokens = ["would", "n't", "'ve"]
        tokens = Tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        sentence = "the jones' house"
        expected_tokens = ["the", "jones", "'", "house"]
        tokens = Tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        sentence = "mr. and mrs. jones, etc., went to, e.g., the store"
        expected_tokens = ["mr.", "and", "mrs.", "jones", ",", "etc.", ",", "went", "to", ",",
                           "e.g.", ",", "the", "store"]
        tokens = Tokenizer.tokenize(sentence)
        assert tokens == expected_tokens
