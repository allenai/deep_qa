class Tokenizer:
    """
    Does really simple tokenization.  NLTK was too slow, so we wrote our own simple tokenizer
    instead.  This just does an initial split(), followed by some heuristic filtering of each
    whitespace-delimited token, separating contractions and punctuation.  We assume lower-cased,
    reasonably well-formed English sentences as input.
    """

    # These are certainly incomplete.  But at least it's a start.
    special_cases = set(['mr.', 'mrs.'])
    contractions = set(["n't", "'s"])
    ending_punctuation = set(['"', "'", '.', ',', ';', ')', ']', '}'])
    beginning_punctuation = set(['"', "'", '(', '[', '{'])

    @classmethod
    def tokenize(cls, sentence):
        fields = sentence.split()
        tokens = []
        for field in fields:  # type: str
            if field in cls.special_cases:
                tokens.append(field)
                continue
            add_at_end = []
            if field and field[0] in cls.beginning_punctuation:
                tokens.append(field[0])
                field = field[1:]
            if field and field[-1] in cls.ending_punctuation:
                add_at_end.insert(0, field[-1])
                field = field[:-1]
            for contraction in cls.contractions:
                if field and field.endswith(contraction):
                    field = field[:-len(contraction)]
                    add_at_end.insert(0, contraction)
                    break
            if field:
                tokens.append(field)
            tokens.extend(add_at_end)
        return tokens
