"""
This module contains the base Instance classes that concrete classes inherit from.  Specifically,
there are three classes:

1. Instance, that just exists as a base type with no functionality
2. TextInstance, which adds a Tokenizer, a words() method, and a method to convert strings to
indices using a DataIndexer.
3. IndexedInstance, which is a TextInstance that has had all of its strings converted into indices.
This class has methods to deal with padding (so that sequences all have the same length) and
converting an Instance into a set of numpy arrays suitable for use with Keras.

As this codebase is dealing mostly with textual question answering, pretty much all of the concrete
Instance types will have both a TextInstance and a corresponding IndexedInstance, which you can see
in the individual files for each Instance type.
"""
from typing import List

from ..data_indexer import DataIndexer
from ..tokenizer import tokenizers, Tokenizer

class Instance:
    """
    A data instance, used either for training a neural network or for testing one.
    """
    def __init__(self, label, index: int=None):
        """
        label: Could be boolean or an index.  For simple Instances (like TextInstance), this is
            either True, False, or None, indicating whether the instance is a positive, negative or
            unknown (i.e., test) example, respectively.  For MultipleChoiceInstances or other more
            complicated things, is a class index.
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
    An Instance that has some attached text, typically either a sentence or a logical form. Calling
    this a "TextInstance" is because the individual tokens here are encoded as strings, and we can
    get a list of strings out when we ask what words show up in the instance.

    We use these kinds of instances to fit a DataIndexer (e.g., deciding which words should be
    mapped to an unknown token); to use them in training or testing, we need to first convert them
    into IndexedInstances.
    """
    def __init__(self,
                 label,
                 index: int=None,
                 tokenizer: Tokenizer=tokenizers['default']()):
        super(TextInstance, self).__init__(label, index)
        self.tokenizer = tokenizer

    def _tokenize(self, sentence: str) -> List[str]:
        """
        Lowercases and then tokenizes the string, using self.tokenizer.
        """
        return self.tokenizer.tokenize(sentence.lower())

    def _index_text(self, text: str, data_indexer: DataIndexer) -> List[int]:
        """
        Tokenizes the given sentence with self._tokenize, then passes the tokens through the
        DataIndexer to get a list of integers out.
        """
        tokens = self._tokenize(text)
        return [data_indexer.get_word_index(token) for token in tokens]

    def words(self) -> List[str]:
        """
        Returns a list of all of the words in this instance.  This is mainly used for computing
        word counts when fitting a word vocabulary on a dataset.
        """
        raise NotImplementedError

    def to_indexed_instance(self, data_indexer: DataIndexer) -> 'IndexedInstance':
        """
        Converts the words in this Instance into indices using the DataIndexer.
        """
        raise NotImplementedError

    @classmethod
    def read_from_line(cls,
                       line: str,
                       default_label: bool=None,
                       tokenizer: Tokenizer=tokenizers['default']()):
        """
        Reads an instance of this type from a line.  We throw a RuntimeError here instead of a
        NotImplementedError, because it's not expected that all subclasses will implement this.
        """
        # pylint: disable=unused-argument
        raise RuntimeError("%s instances can't be read from a line!" % str(cls))


class IndexedInstance(Instance):
    """
    An indexed data instance has all word tokens replaced with word indices, along with some kind
    of label, suitable for input to a Keras model.  An IndexedInstance is created from an Instance
    using a DataIndexer, and the indices here have no recoverable meaning without the DataIndexer.

    For example, we might have the following instance:
        TrueFalseInstance('Jamie is nice, Holly is mean', True, 25).
    After being converted into an IndexedInstance, we might have the following:
        IndexedTrueFalseInstance([1, 6, 7, 1, 6, 8], True, 25).
    This would mean that "Jamie" and "Holly" were OOV to the DataIndexer, and the other words were
    given indices.
    """
    @classmethod
    def empty_instance(cls):
        """
        Returns an empty, unpadded instance of this class.  Necessary for option padding in
        multiple choice instances.
        """
        raise NotImplementedError

    def get_lengths(self) -> List[int]:
        """
        Used for padding.  Different kinds of instances have different fields that are padded, such
        as sentence length, number of background sentences, number of options, etc.  The length of
        this instance in all dimensions that require padding are returned here.
        """
        raise NotImplementedError

    def pad(self, max_lengths: List[int]):
        """
        The max_lengths argument passed here must have the same dimension as was returned by
        get_lengths().  We will use these lengths to pad the instance in all of the necessary
        dimensions to the given lengths.

        This modifies the current object.
        """
        raise NotImplementedError

    def as_training_data(self):
        """
        Returns a tuple of (inputs, label).  `inputs` might itself be a complex tuple, depending on
        the Instance type.
        """
        raise NotImplementedError

    @staticmethod
    def pad_word_sequence_to_length(indices: List[int], desired_length: int) -> List[int]:
        """
        Take a list of word indices and pads them to the desired length.

        If we need to truncate the word indices, we do it from the _right_, not the left.  This is
        important for cases that are questions, with long set ups.  We at least want to get the
        question encoded, which is always at the end, even if we've lost much of the question set
        up.

        Though, this might be backwards for encoding background information...
        """
        padded_indices = [0] * desired_length
        indices_length = min(len(indices), desired_length)
        if indices_length != 0:
            padded_indices[-indices_length:] = indices[-indices_length:]
        return padded_indices
