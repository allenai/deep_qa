# pylint: disable=no-self-use,invalid-name

from dlfa.data.tokenizer import SimpleTokenizer


class TestSimpleTokenizer:
    tokenizer = SimpleTokenizer()
    def test_tokenize_handles_complex_punctuation(self):
        sentence = "this (sentence) has 'crazy' \"punctuation\"."
        expected_tokens = ["this", "(", "sentence", ")", "has", "'", "crazy", "'", '"',
                           "punctuation", '"', "."]
        tokens = self.tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_contraction(self):
        sentence = "it ain't joe's problem; would've been yesterday"
        expected_tokens = ["it", "ai", "n't", "joe", "'s", "problem", ";", "would", "'ve", "been",
                           "yesterday"]
        tokens = self.tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_multiple_contraction(self):
        sentence = "wouldn't've"
        expected_tokens = ["would", "n't", "'ve"]
        tokens = self.tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_final_apostrophe(self):
        sentence = "the jones' house"
        expected_tokens = ["the", "jones", "'", "house"]
        tokens = self.tokenizer.tokenize(sentence)
        assert tokens == expected_tokens

    def test_tokenize_handles_special_cases(self):
        sentence = "mr. and mrs. jones, etc., went to, e.g., the store"
        expected_tokens = ["mr.", "and", "mrs.", "jones", ",", "etc.", ",", "went", "to", ",",
                           "e.g.", ",", "the", "store"]
        tokens = self.tokenizer.tokenize(sentence)
        assert tokens == expected_tokens
