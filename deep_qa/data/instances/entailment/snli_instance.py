from overrides import overrides

from .sentence_pair_instance import SentencePairInstance


class SnliInstance(SentencePairInstance):
    """
    An SnliInstance is a SentencePairInstance that represents a pair of (text, hypothesis) from the
    Stanford Natural Language Inference (SNLI) dataset, with an associated label.  The main thing
    we need to add here is handling of the label, because there are a few different ways we can use
    this Instance.

    The label can either be a three-way decision (one of either "entails", "contradicts", or
    "neutral"), or a binary decision (grouping either "entails" and "contradicts", for relevance
    decisions, or "contradicts" and "neutral", for entails/not entails decisions.

    The input label must be one of the strings in the label_mapping field below.  The difference
    between the ``*_softmax`` and ``*_sigmoid`` labels are just for implementation reasons.  A softmax over
    two dimensions is exactly equivalent to a sigmoid, but to make our lives easier in building
    models, sometimes we use a sigmoid and sometimes we use a softmax over two dimensions.  Having
    separate labels for these cases makes it easier to use this data in whatever kind of model you
    want.

    It might make sense to push this difference more generally into some common place, so that we
    can separate the label itself from how it's encoded for training.  But that might also be
    complicated to implement, and it's not needed right now.  TODO(matt): if we find ourselves
    doing this kind of thing in several places, we should think about making that change.
    """
    label_mapping = {
            "entails": [1, 0, 0],
            "contradicts": [0, 1, 0],
            "neutral": [0, 0, 1],
            "attention_true": [1],
            "attention_false": [0],
            "entails_softmax": [0, 1],
            "not_entails_softmax": [1, 0],
            "entails_sigmoid": [1],
            "not_entails_sigmoid": [0],
            }

    def __init__(self, text: str, hypothesis: str, label: str, index: int=None):
        # This intentionally crashes if `label` is not one of the keys in `label_mapping`.
        super(SnliInstance, self).__init__(text, hypothesis, self.label_mapping[label], index)

    def __str__(self):
        return 'SnliInstance(' + self.first_sentence + ', ' + self.second_sentence + ', ' + str(self.label) + ')'

    def to_attention_instance(self):
        """
        This returns a new SnliInstance with a different label.
        """
        if self.label == self.label_mapping["entails"] or self.label == self.label_mapping["contradicts"]:
            new_label = "attention_true"
        elif self.label == self.label_mapping["neutral"]:
            new_label = "attention_false"
        else:
            raise RuntimeError("Can't convert " + str(self.label) + " to an attention label")
        return SnliInstance(self.first_sentence, self.second_sentence, new_label, self.index)

    def to_entails_instance(self, activation: str):
        """
        This returns a new SnliInstance with a different label.  The new label will be binary
        (entails / not entails), but we need to distinguish between two different label types.
        Sometimes we need the label to be encoded in a single dimension (i.e., either `0` or `1`),
        and sometimes we need it to be encoded in two dimensions (i.e., either `[0, 1]` or `[1,
        0]`).  This depends on the activation function of the final layer in our network - a
        sigmoid activation will need the former, while a softmax activation will need the later.
        So, we encode these differently, as strings, which will be converted to the right array
        later, in IndexedSnliInstance.
        """
        if self.label == self.label_mapping["entails"]:
            new_label = "entails"
        elif self.label == self.label_mapping["neutral"] or self.label == self.label_mapping["contradicts"]:
            new_label = "not_entails"
        else:
            raise RuntimeError("Can't convert " + str(self.label) + " to an entails/not-entails label")
        new_label += '_' + activation
        return SnliInstance(self.first_sentence, self.second_sentence, new_label, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads an SnliInstance object from a line.  The format has one of two options:

        (1) [example index][tab][text][tab][hypothesis][tab][label]
        (2) [text][tab][hypothesis][tab][label]

        [label] is assumed to be one of "entails", "contradicts", or "neutral".
        """
        fields = line.split("\t")

        if len(fields) == 4:
            index_string, text, hypothesis, label = fields
            index = int(index_string)
        elif len(fields) == 3:
            text, hypothesis, label = fields
            index = None
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        return cls(text, hypothesis, label, index)
