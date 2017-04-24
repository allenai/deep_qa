"""
This module contains the base ``Instance`` classes that concrete classes
inherit from. Specifically, there are three classes:

1. ``Instance``, that just exists as a base type with no functionality
2. ``TextInstance``, which adds a ``words()`` method and a method to convert
   strings to indices using a DataIndexer.
3. ``IndexedInstance``, which is a ``TextInstance`` that has had all of its
   strings converted into indices.

This class has methods to deal with padding (so that sequences all have the
same length) and converting an ``Instance`` into a set of Numpy arrays
suitable for use with Keras.

As this codebase is dealing mostly with textual question answering, pretty much
all of the concrete ``Instance`` types will have both a ``TextInstance`` and a
corresponding ``IndexedInstance``, which you can see in the individual files
for each ``Instance`` type.
"""
import itertools
from typing import Any, Callable, Dict, List

from ...common.params import Params
from ..tokenizers import tokenizers
from ..data_indexer import DataIndexer


class Instance:
    """
    A data instance, used either for training a neural network or for testing one.

    Parameters
    ----------
    label : Any
        Any kind of label that you might want to predict in a model.  Could be a class label, a
        tag sequence, a character span in a passage, etc.
    index : int, optional
        Used for matching instances with other data, such as background
        sentences.
    """
    def __init__(self, label, index: int=None):
        self.label = label
        self.index = index


class TextInstance(Instance):
    """
    An ``Instance`` that has some attached text, typically either a sentence
    or a logical form. This is called a ``TextInstance`` because the
    individual tokens here are encoded as strings, and we can
    get a list of strings out when we ask what words show up in the instance.

    We use these kinds of instances to fit a ``DataIndexer`` (i.e., deciding
    which words should be mapped to an unknown token); to use them in training
    or testing, we need to first convert them into ``IndexedInstances``.

    In order to actually convert text into some kind of indexed sequence,
    we rely on a ``TextEncoder``. There are several ``TextEncoder`` subclasses,
    that will let you use word token sequences, character sequences, and other
    options.  By default we use word tokens.  You can override this by setting
    the ``encoder`` class variable.
    """
    tokenizer = tokenizers['words'](Params({}))

    def __init__(self, label, index: int=None):
        super(TextInstance, self).__init__(label, index)

    def _words_from_text(self, text: str) -> Dict[str, List[str]]:
        return self.tokenizer.get_words_for_indexer(text)

    def _index_text(self, text: str, data_indexer: DataIndexer) -> List[int]:
        return self.tokenizer.index_text(text, data_indexer)

    def words(self) -> Dict[str, List[str]]:
        """
        Returns a list of all of the words in this instance, contained in a
        namespace dictionary.

        This is mainly used for computing word counts when fitting a word
        vocabulary on a dataset. The namespace dictionary allows you to have
        several embedding matrices with different vocab sizes, e.g., for words
        and for characters (in fact, words and characters are the only use
        cases I can think of for now, but this allows you to do other more
        crazy things if you want). You can call the namespaces whatever you
        want, but if you want the ``DataIndexer`` to work correctly without
        namespace arguments, you should use the key 'words' to represent word
        tokens.

        Returns
        -------
        namespace : Dictionary of {str: List[str]}
            The ``str`` key refers to vocabularies, and the ``List[str]``
            should contain the tokens in that vocabulary. For example, you
            should use the key ``words`` to represent word tokens, and the
            correspoding value in the dictionary would be a list of all the
            words in the instance.
        """
        raise NotImplementedError

    def to_indexed_instance(self, data_indexer: DataIndexer) -> 'IndexedInstance':
        """
        Converts the words in this ``Instance`` into indices using
        the ``DataIndexer``.

        Parameters
        ----------
        data_indexer : DataIndexer
            ``DataIndexer`` to use in converting the ``Instance`` to
            an ``IndexedInstance``.

        Returns
        -------
        indexed_instance : IndexedInstance
            A ``TextInstance`` that has had all of its strings converted into
            indices.
        """
        raise NotImplementedError

    @classmethod
    def read_from_line(cls, line: str):
        """
        Reads an instance of this type from a line.

        Parameters
        ----------
        line : str
            A line from a data file.

        Returns
        -------
        indexed_instance : IndexedInstance
            A ``TextInstance`` that has had all of its strings converted into
            indices.

        Notes
        -----
        We throw a ``RuntimeError`` here instead of a ``NotImplementedError``,
        because it's not expected that all subclasses will implement this.

        """
        # pylint: disable=unused-argument
        raise RuntimeError("%s instances can't be read from a line!" % str(cls))


class IndexedInstance(Instance):
    """
    An indexed data instance has all word tokens replaced with word indices,
    along with some kind of label, suitable for input to a Keras model. An
    ``IndexedInstance`` is created from an ``Instance`` using a
    ``DataIndexer``, and the indices here have no recoverable meaning without
    the ``DataIndexer``.

    For example, we might have the following ``Instance``:
    - ``TrueFalseInstance('Jamie is nice, Holly is mean', True, 25)``

    After being converted into an ``IndexedInstance``, we might have
    the following:
    - ``IndexedTrueFalseInstance([1, 6, 7, 1, 6, 8], True, 25)``

    This would mean that ``"Jamie"`` and ``"Holly"`` were OOV to the
    ``DataIndexer``, and the other words were given indices.
    """
    @classmethod
    def empty_instance(cls):
        """
        Returns an empty, unpadded instance of this class. Necessary for option
        padding in multiple choice instances.

        """
        raise NotImplementedError

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        Returns the length of this instance in all dimensions that require padding.

        Different kinds of instances have different fields that are padded, such as sentence
        length, number of background sentences, number of options, etc.

        Returns
        -------
        padding_lengths: Dict[str, int]
            A dictionary mapping padding keys (like "num_sentence_words") to lengths.
        """
        raise NotImplementedError

    def pad(self, padding_lengths: Dict[str, int]):
        """
        Add zero-padding to make each data example of equal length for use
        in the neural network.

        This modifies the current object.

        Parameters
        ----------
        padding_lengths: Dict[str, int]
            In this dictionary, each ``str`` refers to a type of token (e.g.
            ``num_sentence_words``), and the corresponding ``int`` is the value. This dictionary
            must have the same keys as was returned by
            :func:`~IndexedInstance.get_padding_lengths()`. We will use these lengths to pad the
            instance in all of the necessary dimensions to the given leangths.
        """
        raise NotImplementedError

    def as_training_data(self):
        """
        Convert this ``IndexedInstance`` to NumPy arrays suitable for use as
        training data to Keras models.

        Returns
        -------
        train_data : (inputs, label)
            The ``IndexedInstance`` as NumPy arrays to be uesd in Keras.
            Note that ``inputs`` might itself be a complex tuple, depending
            on the ``Instance`` type.
        """
        raise NotImplementedError

    @staticmethod
    def _get_word_sequence_lengths(word_indices: List) -> Dict[str, int]:
        """
        Because ``TextEncoders`` can return complex data structures, we might
        actually have several things to pad for a single word sequence. We
        check for that and handle it in a single spot here. We return a
        dictionary containing 'num_sentence_words', which is the number of
        words in word_indices. If the word representations also contain
        characters, the dictionary additionally contains a
        'num_word_characters' key, with a value corresponding to the longest
        word in the sequence.
        """
        padding_lengths = {'num_sentence_words': len(word_indices)}
        if len(word_indices) > 0 and not isinstance(word_indices[0], int):
            if isinstance(word_indices[0], list):
                padding_lengths['num_word_characters'] = max([len(word) for word in word_indices])
            # There might someday be other cases we're missing here, but we'll punt for now.
        return padding_lengths

    @staticmethod
    def pad_word_sequence(word_sequence: List[int],
                          padding_lengths: Dict[str, int],
                          truncate_from_right: bool=True) -> List:
        """
        Take a list of indices and pads them.

        Parameters
        ----------
        word_sequence : List of int
            A list of word indices.

        padding_lengths : Dict[str, int]
            In this dictionary, each ``str`` refers to a type of token (e.g.
            ``num_sentence_words``), and the corresponding ``int`` is the value. This dictionary
            must have the same dimension as was returned by
            :func:`~IndexedInstance.get_padding_lengths()`. We will use these lengths to pad the
            instance in all of the necessary dimensions to the given leangths.

        truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates whether we do so on the
            left or right.

        Returns
        -------
        padded_word_sequence : List of int
            A padded list of word indices.

        Notes
        -----
        The reason we truncate from the right by default is for cases that are questions, with long
        set ups. We at least want to get the question encoded, which is always at the end, even if
        we've lost much of the question set up. If you want to truncate from the other direction,
        you can.

        TODO(matt): we should probably switch the default to truncate from the left, and clear up
        the naming here - it's easy to get confused about what "truncate from right" means.
        """
        default_value = lambda: 0
        if 'num_word_characters' in padding_lengths:
            default_value = lambda: []

        padded_word_sequence = IndexedInstance.pad_sequence_to_length(
                word_sequence, padding_lengths['num_sentence_words'], default_value, truncate_from_right)
        if 'num_word_characters' in padding_lengths:
            desired_length = padding_lengths['num_word_characters']
            longest_word = max(padded_word_sequence, key=len)
            if desired_length > len(longest_word):
                # since we want to pad to greater than the longest word, we add a
                # "dummy word" to get the speed of itertools.zip_longest
                padded_word_sequence.append([0]*desired_length)
            # pad the list of lists to the longest sublist, appending 0's
            words_padded_to_longest = list(zip(*itertools.zip_longest(*padded_word_sequence,
                                                                      fillvalue=0)))
            if desired_length > len(longest_word):
                # now we remove the "dummy word" if we appended one.
                words_padded_to_longest.pop()

            # now we need to truncate all of them to our desired length.
            # since truncate_from_right is always False, we chop off starting from
            # the right.
            padded_word_sequence = [list(word[:desired_length])
                                    for word in words_padded_to_longest]
        return padded_word_sequence

    @staticmethod
    def pad_sequence_to_length(sequence: List,
                               desired_length: int,
                               default_value: Callable[[], Any]=lambda: 0,
                               truncate_from_right: bool=True) -> List:
        """
        Take a list of indices and pads them to the desired length.

        Parameters
        ----------
        word_sequence : List of int
            A list of word indices.

        desired_length : int
            Maximum length of each sequence. Longer sequences
            are truncated to this length, and shorter ones are padded to it.

        default_value: Callable, default=lambda: 0
            Callable that outputs a default value (of any type) to use as
            padding values.

        truncate_from_right : bool, default=True
            If truncating the indices is necessary, this parameter dictates
            whether we do so on the left or right.

        Returns
        -------
        padded_word_sequence : List of int
            A padded or truncated list of word indices.

        Notes
        -----
        The reason we truncate from the right by default is for
        cases that are questions, with long set ups. We at least want to get
        the question encoded, which is always at the end, even if we've lost
        much of the question set up. If you want to truncate from the other
        direction, you can.
        """
        if truncate_from_right:
            truncated = sequence[-desired_length:]
        else:
            truncated = sequence[:desired_length]
        if len(truncated) < desired_length:
            # If the length of the truncated sequence is less than the desired
            # length, we need to pad.
            padding_sequence = [default_value()] * (desired_length - len(truncated))
            if truncate_from_right:
                # When we truncate from the right, we add zeroes to the front.
                padding_sequence.extend(truncated)
                return padding_sequence
            else:
                # When we do not truncate from the right, we add zeroes to the end.
                truncated.extend(padding_sequence)
                return truncated
        return truncated
