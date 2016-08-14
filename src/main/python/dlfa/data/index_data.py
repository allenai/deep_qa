import warnings
from nltk.tokenize import word_tokenize
from .constants import SHIFT_OP, REDUCE2_OP, REDUCE3_OP


class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0, "UNK":1, ",":2, ".":3, "(":4, ")":5}
        self.reverse_word_index = {0:"PADDING", 1:"UNK", 2:",", 3:".", 4:"(", 5:")"}
        # We differentiate between predicate and argument indices for type specific corruption.  If
        # the corruption code ever gets removed from this class, these variables can go away (along
        # with the corresponding logic in process_data).
        self.predicate_indices = set([])
        self.argument_indices = set([])

    def fit_word_dictionary(self, dataset: 'Dataset'):
        """
        Given a Dataset, this method decides which words are given an index, and which ones are
        mapped to an OOV token (in this case "UNK").  This method must be called before any dataset
        is indexed with this DataIndexer.  If you don't first fit the word dictionary, you'll
        basically map every token onto "UNK".

        We use nltk.tokenize.word_tokenize on the text in each Instance in the Dataset, and then
        keep all words that appear at least once.

        If the dataset involves logical forms, we'll also try to keep track of which words are
        predicates and which ones are arguments, so we can do slightly more intelligent data
        corruption if requested.
        """
        predicate_words = set()
        argument_words = set()

        # TODO(matt): might be more efficient and configurable to just keep a count, instead of
        # using these two sets.  Then the caller can set a threshold.
        singletons = set([])
        non_singletons = set([])
        for instance in dataset.instances:
            words = word_tokenize(instance.text.lower())
            for i, word in enumerate(words):
                if i < len(words)-1:
                    next_word = words[i+1]
                    if next_word == "(":
                        # If the next token is an opening paren, this token is a predicate.
                        predicate_words.add(word)
                    elif (next_word == "," or next_word == ")") and word != ")":
                        argument_words.add(word)
                if word not in non_singletons:
                    if word in singletons:
                        # Since we are seeing the word again, it is not a singleton
                        singletons.remove(word)
                        non_singletons.add(word)
                    else:
                        # We have not seen this word. It is a singleton (atleast for now)
                        singletons.add(word)
        for word in non_singletons:
            if word not in self.word_index:
                self.word_index[word] = len(self.word_index)
            if word in predicate_words:
                self.predicate_indices.add(self.word_index[word])
            if word in argument_words:
                self.argument_indices.add(self.word_index[word])
        self.reverse_word_index = {index:word for (word, index) in self.word_index.items()}

    def get_word_index(self, word: str):
        if word in self.word_index:
            return self.word_index[word]
        else:
            return self.word_index["UNK"]

    def get_shift_reduce_sequences(self, word_arrays):
        """
        This function splits sequences containing parantheses and commas into two sequences, one
        containing only elements (predicates and arguments) and the other containing shift and
        reduce operations.

        Example:
        Input: ['a', '(', 'b', '(', 'c', ')', ',', 'd', '(', 'e', ',', 'f', ')', ')']
        Outputs: ['a', 'b', 'c', 'd', 'e', 'f']; [S, S, S, R2, S, S, S, R3, R3]

        Note that this function operates on the output of IndexedDataset.as_training_data.
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
