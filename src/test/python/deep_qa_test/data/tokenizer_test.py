# pylint: disable=no-self-use,invalid-name

from deep_qa.data.tokenizer import SimpleTokenizer


class TestTokenizer:
    tokenizer = SimpleTokenizer()
    passage = "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy " +\
        "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four " +\
        "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her " +\
        "first performances since giving birth to Blue Ivy."

    def test_char_span_to_token_span_handles_easy_cases(self):
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (3, 18))
        assert token_span == (1, 4)
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (91, 110))
        assert token_span == (22, 24)
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (91, 123))
        assert token_span == (22, 28)


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
