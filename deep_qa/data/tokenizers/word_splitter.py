from collections import OrderedDict
from typing import List

from overrides import overrides


class WordSplitter:
    """
    A ``WordSplitter`` splits strings into words.  This is typically called a "tokenizer" in NLP,
    but we need ``Tokenizer`` to refer to something else, so we're using ``WordSplitter`` here
    instead.
    """

    def split_words(self, sentence: str) -> List[str]:
        raise NotImplementedError


class SimpleWordSplitter(WordSplitter):
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
    def split_words(self, sentence: str) -> List[str]:
        """
        Splits a sentence into word tokens.  We handle four kinds of things: words with punctuation
        that should be ignored as a special case (Mr. Mrs., etc.), contractions/genitives (isn't,
        don't, Matt's), and beginning and ending punctuation ("antennagate", (parentheticals), and
        such.).

        The basic outline is to split on whitespace, then check each of these cases.  First, we
        strip off beginning punctuation, then strip off ending punctuation, then strip off
        contractions.  When we strip something off the beginning of a word, we can add it to the
        list of tokens immediately.  When we strip it off the end, we have to save it to be added
        to after the word itself has been added.  Before stripping off any part of a token, we
        first check to be sure the token isn't in our list of special cases.
        """
        fields = sentence.lower().split()
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


class NltkWordSplitter(WordSplitter):
    """
    A tokenizer that uses nltk's word_tokenize method.

    I found that nltk is very slow, so I switched to using my own simple one, which is a good deal
    faster.  But I'm adding this one back so that there's consistency with older versions of the
    code, if you really want it.
    """
    @overrides
    def split_words(self, sentence: str) -> List[str]:
        # Import is here because it's slow, and by default unnecessary.
        from nltk.tokenize import word_tokenize
        return word_tokenize(sentence.lower())


class NoOpWordSplitter(WordSplitter):
    """
    This is a word splitter that does nothing.  We're playing a little loose with python's dynamic
    typing, breaking the typical WordSplitter API a bit and assuming that you've already split
    ``sentence`` into a list somehow, so you don't need to do anything else here.  For example, the
    ``PreTokenizedTaggingInstance`` requires this word splitter, because it reads in pre-tokenized
    data from a file.
    """
    @overrides
    def split_words(self, sentence: str) -> List[str]:
        assert isinstance(sentence, list), "This splitter is only meant to be used for pre-split text"
        return sentence


word_splitters = OrderedDict()  # pylint: disable=invalid-name
word_splitters['simple'] = SimpleWordSplitter
word_splitters['nltk'] = NltkWordSplitter
word_splitters['no_op'] = NoOpWordSplitter
