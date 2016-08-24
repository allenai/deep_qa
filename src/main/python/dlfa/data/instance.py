from typing import List

from .constants import SHIFT_OP, REDUCE2_OP, REDUCE3_OP
from .indexed_instance import IndexedInstance, IndexedBackgroundInstance, IndexedLogicalFormInstance
from .tokenizer import Tokenizer
from .data_indexer import DataIndexer

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
    An Instance that has some attached text, typically either a sentence or a logical form. Calling
    this a "TextInstance" is because the individual tokens here are encoded as strings, and we can
    get a list of strings out when we ask what words show up in the instance.

    We use these kinds of instances to fit a DataIndexer (e.g., deciding which words should be
    mapped to an unknown token); to use them in training or testing, we need to first convert them
    into IndexedInstances.
    """
    @staticmethod
    def tokenize(sentence: str, tokenizer=Tokenizer()) -> List[str]:
        return tokenizer.tokenize(sentence)

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

    def words(self) -> List[str]:
        return self.tokenize(self.text.lower())

    def to_indexed_instance(self, data_indexer: DataIndexer):
        indices = [data_indexer.get_word_index(word) for word in self.words()]
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


class LogicalFormInstance(TextInstance):
    """
    This is an instance for use with TreeLSTMs.  Instead of a sequence of words, this Instance is a
    tree-structured logical form, encoded as something like "for(depend_on(human, plant), oxygen)".

    We base this off of TextInstance because we implement the same interface, though we override
    most of the methods.
    """
    def words(self) -> List[str]:
        """
        This method takes all predicate names and arguments and returns them, removing commas and
        parentheses.
        """
        return [word for word in self.tokens() if word != ',' and word != ')' and word != '(']

    def tokens(self) -> List[str]:
        """
        This method splits the logical form into tokens, including commas and parentheses.
        """
        raise NotImplementedError

    def to_indexed_instance(self, data_indexer: DataIndexer):
        """
        Here we have to split the logical forms (containing parantheses and commas) into two
        sequences, one containing only elements (predicates and arguments) and the other containing
        shift and reduce operations.

        Example:
        self.text: "a(b(c), d(e, f))"
        Outputs: ['a', 'b', 'c', 'd', 'e', 'f']; [S, S, S, R2, S, S, S, R3, R3]

        (except the real output passes the elements through the data indexer as well)
        """
        last_symbols = []  # Keeps track of commas and open parens
        transitions = []
        elements = []
        is_malformed = False
        for token in self.tokens():
            if token == ',':
                last_symbols.append(token)
            elif token == '(':
                last_symbols.append(token)
            elif token == ')':
                if len(last_symbols) == 0:
                    # This means we saw a closing paren without an opening paren.
                    is_malformed = True
                    break
                last_symbol = last_symbols.pop()
                if last_symbol == '(':
                    transitions.append(REDUCE2_OP)
                else:
                    # Last symbol is a comma. Pop the open paren before it as well.
                    last_symbols.pop()
                    transitions.append(REDUCE3_OP)
            else:
                # The token is a predicate or an argument.
                transitions.append(SHIFT_OP)
                elements.append(token)
        if len(last_symbols) != 0 or is_malformed:
            raise RuntimeError("Malformed binary semantic parse: %s" % self.text)
        indices = [data_indexer.get_word_index(word) for word in elements]
        return IndexedLogicalFormInstance(indices, transitions, self.label, self.index)


class BackgroundTextInstance(TextInstance):
    """
    An Instance that has background knowledge associated with it.

    TODO(matt): it might make sense to have `background` be a list of instances, instead of a list
    of strings, to allow for easier combination of background instances with logical form
    instances.  But, we'll worry about that later, if it ever becomes important.
    """
    def __init__(self, text: str, background: List[str], label: bool, index: int=None):
        super(BackgroundTextInstance, self).__init__(text, label, index)
        self.background = background

    def words(self):
        text_words = super(BackgroundTextInstance, self).words()
        background_words = []
        for background_text in self.background:
            background_words.extend(self.tokenize(background_text.lower()))
        text_words.extend(background_words)
        return text_words

    def to_indexed_instance(self, data_indexer: DataIndexer):
        words = self.tokenize(self.text.lower())
        word_indices = [data_indexer.get_word_index(word) for word in words]
        background_indices = []
        for text in self.background:
            words = self.tokenize(text.lower())
            indices = [data_indexer.get_word_index(word) for word in words]
            background_indices.append(indices)
        return IndexedBackgroundInstance(word_indices, background_indices, self.label, self.index)
