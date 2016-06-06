import random
from nltk.tokenize import word_tokenize

class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0, "UNK":1}
        self.singletons = set([])
        self.non_singletons = set([])

    def get_indices(self, lines, pad=True, separate_propositions=True, for_train=True):
        '''
        lines: list(str). Sequence of proposition strings. If propositions have to be grouped (to keep track of which sentences they come from), separate props in a line with ';'
        pad: bool. Left pad index vectors to make them all the same size?
        separate_propositions: bool. Use ';' as a proposition delimiter?
        for_train: bool. Setting this to False will treat new words as OOV. Setting it to True will add them to the index if they occur atleast twice. Singletons will always be treated as OOV.
        '''
        all_proposition_indices = []
        num_propositions_in_lines = []
        all_proposition_words = []
        for line in lines:
            if separate_propositions:
                line_propositions = line.split(";")
            else:
                line_propositions = [line]
            num_propositions_in_lines.append(len(line_propositions))
            for proposition in line_propositions:
                words = word_tokenize(proposition.lower())
                all_proposition_words.append(words)
                if for_train:
                    for word in words:
                        if word not in self.non_singletons:
                            if word in self.singletons:
                                # Since we are seeing the word again, it is not a singleton
                                self.singletons.remove(word)
                                self.non_singletons.add(word)
                            else:
                                # We have not seen this word. It is a singleton (atleast for now)
                                self.singletons.add(word)
        if for_train:
            # Add non-singletons to index. The singletons will remain unknown.
            for word in self.non_singletons:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)

        for words in all_proposition_words:
            proposition_indices = [self.word_index[word] if word in self.word_index else self.word_index["UNK"] for word in words]
            all_proposition_indices.append(proposition_indices)

        if pad:
            return num_propositions_in_lines, self.pad_indices(all_proposition_indices)
        else:
            return num_propositions_in_lines, all_proposition_indices

    def pad_indices(self, all_indices, max_length=None):
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
        # We would want to ignore the padding token, parentheses and commas while corrputing data
        indices_to_ignore = set([0])
        for token in [",", "(", ")", "."]:
            if token in self.word_index:
                indices_to_ignore.add(self.word_index[token])
        for indices in all_indices:
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
                rand_index = 0
                while rand_index in indices_to_ignore:
                    rand_index = random.randint(1, len(self.word_index)-1)
                corrupted_indices[rand_location] = rand_index
                num_corrupted_locations += 1
            all_corrupted_indices.append(corrupted_indices)
        return all_corrupted_indices
