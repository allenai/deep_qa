import warnings

from collections import defaultdict

from .constants import SHIFT_OP, REDUCE2_OP, REDUCE3_OP


class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0, "UNK":1, ",":2, ".":3, "(":4, ")":5}
        self.reverse_word_index = {0:"PADDING", 1:"UNK", 2:",", 3:".", 4:"(", 5:")"}

    def fit_word_dictionary(self, dataset: 'Dataset', min_count: int=1):
        """
        Given a Dataset, this method decides which words are given an index, and which ones are
        mapped to an OOV token (in this case "UNK").  This method must be called before any dataset
        is indexed with this DataIndexer.  If you don't first fit the word dictionary, you'll
        basically map every token onto "UNK".

        We call instance.words() for each instance in the dataset, and then keep all words that
        appear at least min_count times.
        """
        word_counts = defaultdict(int)
        for instance in dataset.instances:
            for word in instance.words():
                word_counts[word] += 1
        for word, count in word_counts.items():
            if count > min_count and word not in self.word_index:
                self.word_index[word] = len(self.word_index)
        self.reverse_word_index = {index:word for word, index in self.word_index.items()}

    def words_in_index(self):
        return self.word_index.keys()

    def get_word_index(self, word: str):
        if word in self.word_index:
            return self.word_index[word]
        else:
            return self.word_index["UNK"]

    def get_word_from_index(self, index: int):
        return self.reverse_word_index[index]

    def get_shift_reduce_sequences(self, word_arrays):
        """
        This function splits sequences containing parantheses and commas into two sequences, one
        containing only elements (predicates and arguments) and the other containing shift and
        reduce operations.

        Example:
        Input: ['a', '(', 'b', '(', 'c', ')', ',', 'd', '(', 'e', ',', 'f', ')', ')']
        Outputs: ['a', 'b', 'c', 'd', 'e', 'f']; [S, S, S, R2, S, S, S, R3, R3]

        Note that this function operates on the output of IndexedDataset.as_training_data.

        TODO(matt): this code belongs in the as_training_data method of a new type of Instance, not
        in the DataIndexer.  When we start working with the TreeLSTMs again, this code should be
        moved.
        """
        all_transitions = []
        all_elements = []
        comma_index = self.word_index[","]
        open_paren_index = self.word_index["("]
        close_paren_index = self.word_index[")"]
        for word_array in word_arrays:
            last_symbols = [] # Keeps track of commas and open parens
            transitions = []
            elements = []
            is_malformed = False
            for ind in word_array:
                if ind == comma_index:
                    last_symbols.append(",")
                elif ind == open_paren_index:
                    last_symbols.append("(")
                elif ind == close_paren_index:
                    if len(last_symbols) == 0:
                        # This means we saw a closing paren without an opening paren.
                        # Ignore this parse.
                        is_malformed = True
                        break
                    last_symbol = last_symbols.pop()
                    if last_symbol == "(":
                        transitions.append(REDUCE2_OP)
                    else:
                        # Last symbol is a comma. Pop the open paren
                        # before it as well.
                        last_symbols.pop()
                        transitions.append(REDUCE3_OP)
                else:
                    # The token is padding, predicate or an argument.
                    transitions.append(SHIFT_OP)
                    elements.append(ind)
            if len(last_symbols) != 0 or is_malformed:
                # We either have more opening parens than closing parens, or we
                # ignored the parse earlier. Throw a warning, and ignore this parse.
                parse = self.get_words_from_indices(word_array)
                warnings.warn("Malformed binary semantic parse: %s" % parse, RuntimeWarning)
                all_transitions.append([])
                all_elements.append([])
                continue
            all_transitions.append(transitions)
            all_elements.append(elements)
        return all_transitions, all_elements

    def get_vocab_size(self):
        return len(self.word_index) + 1

    def get_words_from_indices(self, indices):
        return " ".join([self.reverse_word_index[i] for i in indices]).replace("PADDING", "").strip()
