from collections import OrderedDict
from typing import List, Tuple

from overrides import overrides

class Tokenizer:
    """
    A Tokenizer splits strings into tokens.
    """
    def tokenize(self, sentence: str) -> List[str]:
        raise NotImplementedError

    def char_span_to_token_span(self,
                                sentence: str,
                                span: Tuple[int, int],
                                tokenized_sentence: List[str]=None,
                                slack: int=3) -> Tuple[int, int]:
        """
        Converts a character span from a sentence into the corresponding token span in the
        tokenized version of the sentence.  If you pass in a character span that does not
        correspond to complete tokens in the tokenized version, we'll do our best, but the behavior
        is officially undefined.

        If you have already tokenized the sentence, you can pass it in as an argument, to save some
        time.  Otherwise, we'll tokenize the sentence here.

        The basic outline of this method is to find the token that starts the same number of
        characters into the sentence as the given character span.  We try to handle a bit of error
        in the tokenization by checking `slack` tokens in either direction from that initial
        estimate.
        """
        # First we'll tokenize the span and the sentence, so we can count tokens and check for
        # matches.
        span_chars = sentence[span[0]:span[1]]
        tokenized_span = self.tokenize(span_chars)
        if tokenized_sentence is None:
            tokenized_sentence = self.tokenize(sentence)
        # Then we'll find what we think is the first token in the span
        chars_seen = 0
        index = 0
        while index < len(tokenized_sentence) and chars_seen < span[0]:
            chars_seen += len(tokenized_sentence[index]) + 1
            index += 1
        # index is now the span start index.  Is it a match?
        if self._spans_match(tokenized_sentence, tokenized_span, index):
            return (index, index + len(tokenized_span) - 1)
        for i in range(1, slack + 1):
            if self._spans_match(tokenized_sentence, tokenized_span, index + i):
                return (index + i, index + i+ len(tokenized_span) - 1)
            if self._spans_match(tokenized_sentence, tokenized_span, index - i):
                return (index - i, index - i + len(tokenized_span) - 1)
        # No match; we'll just return our best guess.
        return (index, index + len(tokenized_span) - 1)

    @staticmethod
    def _spans_match(sentence_tokens: List[str], span_tokens: List[str], index: int) -> bool:
        if index < 0 or index >= len(sentence_tokens):
            return False
        if sentence_tokens[index] == span_tokens[0]:
            span_index = 1
            while (span_index < len(span_tokens) and
                   sentence_tokens[index + span_index] == span_tokens[span_index]):
                span_index += 1
            if span_index == len(span_tokens):
                return True
        return False



class SimpleTokenizer(Tokenizer):
    """
    Does really simple tokenization.  NLTK was too slow, so we wrote our own simple tokenizer
    instead.  This just does an initial split(), followed by some heuristic filtering of each
    whitespace-delimited token, separating contractions and punctuation.  We assume lower-cased,
    reasonably well-formed English sentences as input.
    """

    # These are certainly incomplete.  But at least it's a start.
    special_cases = set(['mr.', 'mrs.', 'etc.', 'e.g.', 'cf.', 'c.f.', 'eg.', 'al.'])
    contractions = set(["n't", "'s", "'ve", "'re", "'ll", "'d", "'m"])
    contractions |= set([x.replace("'", "’") for x in contractions])
    ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}', ':', '!', '?', '%', '”', "’"])
    beginning_punctuation = set(['"', "'", '(', '[', '{', '#', '$', '“', "‘"])

    @overrides
    def tokenize(self, sentence: str) -> List[str]:
        """
        Splits a sentence into tokens.  We handle four kinds of things: words with punctuation that
        should be ignored as a special case (Mr. Mrs., etc.), contractions/genitives (isn't, don't,
        Matt's), and beginning and ending punctuation ("antennagate", (parentheticals), and such.).

        The basic outline is to split on whitespace, then check each of these cases.  First, we
        strip off beginning punctuation, then strip off ending punctuation, then strip off
        contractions.  When we strip something off the beginning of a word, we can add it to the
        list of tokens immediately.  When we strip it off the end, we have to save it to be added
        to after the word itself has been added.  Before stripping off any part of a token, we
        first check to be sure the token isn't in our list of special cases.
        """
        fields = sentence.split()
        tokens = []
        for field in fields:  # type: str
            add_at_end = []
            while self._can_split(field) and field[0] in self.beginning_punctuation:
                tokens.append(field[0])
                field = field[1:]
            while self._can_split(field) and field[-1] in self.ending_punctuation:
                add_at_end.insert(0, field[-1])
                field = field[:-1]

            # There could (rarely) be several contractions in a word, but we check contractions
            # sequentially, in a random order.  If we've removed one, we need to check again to be
            # sure there aren't others.
            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in self.contractions:
                    if self._can_split(field) and field.endswith(contraction):
                        field = field[:-len(contraction)]
                        add_at_end.insert(0, contraction)
                        remove_contractions = True
            if field:
                tokens.append(field)
            tokens.extend(add_at_end)
        return tokens

    def _can_split(self, token: str):
        return token and token not in self.special_cases


class NltkTokenizer(Tokenizer):
    """
    A tokenizer that uses nltk's word_tokenize method.

    I found that nltk is very slow, so I switched to using my own simple one, which is a good deal
    faster.  But I'm adding this one back so that there's consistency with older versions of the
    code, if you really want it.
    """
    @overrides
    def tokenize(self, sentence: str) -> List[str]:
        # Import is here because it's slow, and by default unnecessary.
        from nltk.tokenize import word_tokenize
        return word_tokenize(sentence)

tokenizers = OrderedDict()  # pylint: disable=invalid-name
tokenizers['default'] = SimpleTokenizer
tokenizers['simple'] = SimpleTokenizer
tokenizers['nltk'] = NltkTokenizer
