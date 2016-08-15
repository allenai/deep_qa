from typing import List

from nltk.tokenize import word_tokenize

from .indexed_instance import IndexedInstance, IndexedBackgroundInstance
from .index_data import DataIndexer

class Instance:
    """
    A data instance, used either for training a neural network or for testing one.
    """
    def __init__(self, label: bool, index: int=None):
        """
        label: True, False, or None, indicating whether the instance is a positive, negative or
            unknown (i.e., test) example, respectively.
        index: if given, must be an integer.  Used for matching instances with other data, such as
            background sentences.
        """
        self.label = label
        self.index = index

    @staticmethod
    def _check_label(label: bool, default_label: bool):
        if default_label is not None and label is not None and label != default_label:
            raise RuntimeError("Default label given with file, and label in file doesn't match!")


class TextInstance(Instance):
    """
    An Instance that has some attached text, typically either a sentence or a logical form.

    The only thing we can do with this kind of Instance is convert it into another kind that is
    actually usable for training / testing.
    """
    def __init__(self, text: str, label: bool, index: int=None):
        """
        text: the text of this instance, either a sentence or a logical form.
        label: True, False, or None, indicating whether the instance is a positive, negative or
            unknown (i.e., test) example, respectively.
        index: if given, must be an integer.  Used for matching instances with other data, such as
            background sentences.
        """
        super(TextInstance, self).__init__(label, index)
        self.text = text

    def to_indexed_instance(self, data_indexer: DataIndexer):
        words = word_tokenize(self.text.lower())
        indices = [data_indexer.get_word_index(word) for word in words]
        return IndexedInstance(indices, self.label, self.index)

    @staticmethod
    def read_from_line(line: str, default_label: bool=None):
        """
        Reads an Instance object from a line.  The format has one of four options:

        (1) [sentence]
        (2) [sentence index][tab][sentence]
        (3) [sentence][tab][label]
        (4) [sentence index][tab][sentence][tab][label]

        For options (1) and (2), we use the default_label to give the Instance a label, and for
        options (3) and (4), we check that default_label matches the label in the file, if
        default_label is given.

        The reason we check for a match between the read label and the default label in cases (3)
        and (4) is that if you passed a default label, you should be confident that everything
        you're reading has that label.  If we find one that doesn't match, you probably messed up
        some parameters somewhere else in your code.
        """
        fields = line.split("\t")

        # We'll call Instance._check_label for all four cases, even though it means passing None to
        # two of them.  We do this mainly for consistency, and in case the _check_label() ever
        # changes to actually do something with the label=None case.
        if len(fields) == 3:
            index, text, label_string = fields
            label = label_string == '1'
            Instance._check_label(label, default_label)
            return TextInstance(text, label, int(index))
        elif len(fields) == 2:
            if fields[0].isdecimal():
                index, text = fields
                Instance._check_label(None, default_label)
                return TextInstance(text, default_label, int(index))
            elif fields[1].isdecimal():
                text, label_string = fields
                label = label_string == '1'
                Instance._check_label(label, default_label)
                return TextInstance(text, label)
            else:
                raise RuntimeError("Unrecognized line format: " + line)
        elif len(fields) == 1:
            text = fields[0]
            Instance._check_label(None, default_label)
            return TextInstance(text, default_label)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class BackgroundTextInstance(TextInstance):
    """
    An Instance that has background knowledge associated with it.
    """
    def __init__(self, text: str, background: List[str], label: bool, index: int=None):
        super(BackgroundTextInstance, self).__init__(text, label, index)
        self.background = background

    def to_indexed_instance(self, data_indexer: DataIndexer):
        words = word_tokenize(self.text.lower())
        word_indices = [data_indexer.get_word_index(word) for word in words]
        background_indices = []
        for text in self.background:
            words = word_tokenize(text.lower())
            indices = [data_indexer.get_word_index(word) for word in words]
            background_indices.append(indices)
        return IndexedBackgroundInstance(word_indices, background_indices, self.label, self.index)
