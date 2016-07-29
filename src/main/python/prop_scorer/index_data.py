import random
import warnings
from nltk.tokenize import word_tokenize
from constants import SHIFT_OP, REDUCE2_OP, REDUCE3_OP

class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0, "UNK":1, ",":2, ".":3, "(":4, ")":5}
        self.reverse_word_index = {0:"PADDING", 1:"UNK", 2:",", 3:".", 4:"(", 5:")"}
        self.singletons = set([])
        self.non_singletons = set([])
        # We differentiate between predicate and argument indices for type specific corruption.  If
        # the corruption code ever gets removed from this class, these variables can go away (along
        # with the corresponding logic in process_data).
        self.predicate_indices = set([])
        self.argument_indices = set([])

    def process_data(self, lines, for_train=True, max_length=None):
        '''
        lines: list(str). Sequence of proposition strings, one per line.
        for_train: bool. Setting this to False will treat new words as OOV.
            Setting it to True will add them to the index if they occur
            atleast twice. Singletons will always be treated as OOV.
        max_length: int. Ignore propositions greater than this length.
            Applicable only during training.
        '''
        all_proposition_indices = []
        all_proposition_words = []
        # We keep track of both predicate and argument words because some words
        # can occur in both sets
        predicate_words = set()
        argument_words = set()
        for proposition in lines:
            words = word_tokenize(proposition.lower())
            if for_train:
                if max_length is not None:
                    if len(words) > max_length:
                        continue
                for i, word in enumerate(words):
                    if i < len(words)-1:
                        next_word = words[i+1]
                        if next_word == "(":
                            # If the next token is an opening paren,
                            # this token is a predicate.
                            predicate_words.add(word)
                        elif (next_word == "," or next_word == ")") and word != ")":
                            argument_words.add(word)
                    if word not in self.non_singletons:
                        if word in self.singletons:
                            # Since we are seeing the word again,
                            #it is not a singleton
                            self.singletons.remove(word)
                            self.non_singletons.add(word)
                        else:
                            # We have not seen this word. It is
                            #a singleton (atleast for now)
                            self.singletons.add(word)
            all_proposition_words.append(words)
        if for_train:
            # Add non-singletons to index. The singletons will remain unknown.
            for word in self.non_singletons:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
                if word in predicate_words:
                    self.predicate_indices.add(self.word_index[word])
                if word in argument_words:
                    self.argument_indices.add(self.word_index[word])

        for words in all_proposition_words:
            proposition_indices = [self.word_index[word]
                                   if word in self.word_index
                                   else self.word_index["UNK"]
                                   for word in words]
            all_proposition_indices.append(proposition_indices)

        # Update the reverse index
        self.reverse_word_index = {index:word for (word, index) in self.word_index.items()}
        return all_proposition_indices

    def get_shift_reduce_sequences(self, all_indices):
        # This function splits sequences containing parantheses and commas into two sequences, one
        # containing only elements (predicates and arguments) and the other containing
        # shift and reduce operations.
        # Example:
        # Input: ['a', '(', 'b', '(', 'c', ')', ',', 'd', '(', 'e', ',', 'f', ')', ')']
        # Outputs: ['a', 'b', 'c', 'd', 'e', 'f']; [S, S, S, R2, S, S, S, R3, R3]
        # Note: This function operates on the indices instead of the actual strings
        all_transitions = []
        all_elements = []
        comma_index = self.word_index[","]
        open_paren_index = self.word_index["("]
        close_paren_index = self.word_index[")"]
        for indices in all_indices:
            last_symbols = [] # Keeps track of commas and open parens
            transitions = []
            elements = []
            is_malformed = False
            for ind in indices:
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
                parse = self.get_words_from_indices(indices)
                warnings.warn("Malformed binary semantic parse: %s"%parse, RuntimeWarning)
                continue
            all_transitions.append(transitions)
            all_elements.append(elements)
        return all_transitions, all_elements

    @staticmethod
    def pad_indices(all_indices, max_length=None):
        if max_length is None:
            max_length = max([len(ind) for ind in all_indices])
        all_padded_indices = []
        for indices in all_indices:
            padded_indices = [0]*max_length
            indices_length = min(len(indices), max_length)
            if indices_length != 0:
                padded_indices[-indices_length:] = indices[-indices_length:]
            all_padded_indices.append(padded_indices)
        return all_padded_indices

    def corrupt_indices(self, all_indices, num_locations_to_corrupt=1):
        all_corrupted_indices = []
        # We would want to ignore the padding token, parentheses and commas
        # while corrputing data
        indices_to_ignore = set([0])
        for token in [",", "(", ")", "."]:
            if token in self.word_index:
                indices_to_ignore.add(self.word_index[token])
        open_paren_index = self.word_index["("]
        for indices in all_indices:
            # Get the first non zero index to avoid sampling from padding
            first_non_zero_index = 0
            for index in indices:
                if index == 0:
                    first_non_zero_index += 1
                else:
                    break
            corrupted_indices = list(indices)
            num_corrupted_locations = 0
            while num_corrupted_locations < num_locations_to_corrupt:
                rand_location = random.randint(first_non_zero_index, len(indices)-1)
                if corrupted_indices[rand_location] in indices_to_ignore:
                    continue
                # If the word after the one being replaced is a "(", the one
                # being replaced is a predicate. We replace it with another predicate
                # If not, it is an argument.
                # Since predicate and argument indices sets contain non-singleton
                # words that are not "(", ")" and ",", we do not need to do other checks.
                next_word_index = -1
                # It is possible the proposition is just one word. In that case,
                # rand_location is the index of that word itself, and we assume
                # the word is a predicate.
                if rand_location < len(corrupted_indices)-1:
                    next_word_index = corrupted_indices[rand_location + 1]
                if next_word_index == open_paren_index or next_word_index == -1:
                    rand_index = random.sample(self.predicate_indices, 1)[0]
                else:
                    rand_index = random.sample(self.argument_indices, 1)[0]
                corrupted_indices[rand_location] = rand_index
                num_corrupted_locations += 1
            all_corrupted_indices.append(corrupted_indices)
        return all_corrupted_indices

    def get_vocab_size(self):
        return len(self.word_index) + 1

    def get_words_from_indices(self, indices):
        return " ".join([self.reverse_word_index[i] for i in indices]).replace("PADDING", "").strip()
