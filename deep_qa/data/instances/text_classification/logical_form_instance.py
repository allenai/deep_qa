from typing import Dict, List

import numpy
from overrides import overrides

from .text_classification_instance import TextClassificationInstance, IndexedTextClassificationInstance
from ...data_indexer import DataIndexer

# Shift and reduce operations in our transition based composition
# We have two kinds of reduce operations:
# REDUCE2: predictate-argument
# REDUCE3: predicate-argument1-argument2
SHIFT_OP = 1
REDUCE2_OP = 2
REDUCE3_OP = 3

class LogicalFormInstance(TextClassificationInstance):
    """
    A LogicalFormInstance is a TextClassificationInstance where the statement is a logical statement.  We
    use these instances with TreeLSTMs.  Statements are assumed to be encoded as something like
    "for(depend_on(human, plant), oxygen)".
    """
    @overrides
    def words(self) -> List[str]:
        """
        This method takes all predicate names and arguments and returns them, removing commas and
        parentheses.
        """
        return {'words': [word for word in self.tokens() if word != ',' and word != ')' and word != '(']}

    def tokens(self) -> List[str]:
        """
        This method splits the logical form into tokens, including commas and parentheses.
        """
        raise NotImplementedError

    @overrides
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


class IndexedLogicalFormInstance(IndexedTextClassificationInstance):
    """
    An IndexedLogicalFormInstance is a tree-structured instance, which represents a logical form
    like "for(depend_on(human, plant), oxygen)" as a pair of: (1) a (sequential) list of predicates
    and arguments, and (2) a list of shift/reduce operations, which allows recovery of the original
    tree structure from the sequential list of predicates and arguments.  This allows us to do tree
    composition in a compiled neural network - we just have to pad to the maximum transition
    length, and we can represent arbitrarily shaped trees.

    Idea taken from the SPINN paper by Sam Bowman and others (http://arxiv.org/pdf/1603.06021.pdf).
    """
    def __init__(self, word_indices: List[int], transitions: List[int], label: bool, index: int=None):
        super(IndexedLogicalFormInstance, self).__init__(word_indices, label, index)
        self.transitions = transitions

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedLogicalFormInstance([], [], None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for `transitions`.
        """
        lengths = super(IndexedLogicalFormInstance, self).get_padding_lengths()
        lengths['transition_length'] = len(self.transitions)
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        We let the super class deal with padding word_indices; we'll worry about padding
        transitions.
        """
        super(IndexedLogicalFormInstance, self).pad(padding_lengths)

        transition_length = padding_lengths['transition_length']
        self.transitions = self.pad_sequence_to_length(self.transitions, transition_length)

    @overrides
    def as_training_data(self):
        word_array, label = super(IndexedLogicalFormInstance, self).as_training_data()
        transitions = numpy.asarray(self.transitions, dtype='int32')
        return (word_array, transitions), label
