# pylint: disable=no-self-use,invalid-name

from deep_qa.data.tokenizers.word_tokenizer import WordTokenizer


class TestTokenizer:
    tokenizer = WordTokenizer({})
    passage = "On January 7, 2012, Beyoncé gave birth to her first child, a daughter, Blue Ivy " +\
        "Carter, at Lenox Hill Hospital in New York. Five months later, she performed for four " +\
        "nights at Revel Atlantic City's Ovation Hall to celebrate the resort's opening, her " +\
        "first performances since giving birth to Blue Ivy."

    def test_char_span_to_token_span_handles_easy_cases(self):
        # "January 7, 2012"
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (3, 18))
        assert token_span == (1, 5)
        # "Lenox Hill Hospital"
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (91, 110))
        assert token_span == (22, 25)
        # "Lenox Hill Hospital in New York."
        token_span = self.tokenizer.char_span_to_token_span(self.passage, (91, 123))
        assert token_span == (22, 29)
