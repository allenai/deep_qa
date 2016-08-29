class Tokenizer:
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

    @classmethod
    def tokenize(cls, sentence: str):
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
            while cls._can_split(field) and field[0] in cls.beginning_punctuation:
                tokens.append(field[0])
                field = field[1:]
            while cls._can_split(field) and field[-1] in cls.ending_punctuation:
                add_at_end.insert(0, field[-1])
                field = field[:-1]

            # There could (rarely) be several contractions in a word, but we check contractions
            # sequentially, in a random order.  If we've removed one, we need to check again to be
            # sure there aren't others.
            remove_contractions = True
            while remove_contractions:
                remove_contractions = False
                for contraction in cls.contractions:
                    if cls._can_split(field) and field.endswith(contraction):
                        field = field[:-len(contraction)]
                        add_at_end.insert(0, contraction)
                        remove_contractions = True
            if field:
                tokens.append(field)
            tokens.extend(add_at_end)
        return tokens

    @classmethod
    def _can_split(cls, token: str):
        return token and token not in cls.special_cases
