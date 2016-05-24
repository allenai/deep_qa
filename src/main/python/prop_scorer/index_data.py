import random
from nltk.tokenize import word_tokenize

class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0}

    def get_indices(self, lines, pad=True, separate_propositions=True):
	'''
	lines: list(str). Sequence of proposition strings. If propositions have to be grouped (to keep track of which sentences they come from), separate props in a line with ';'
	pad: bool. Left pad index vectors to make them all the same size?
	separate_propositions: Use ';' as a proposition delimiter?
	'''
        indices = []
        num_propositions_in_lines = []
        for line in lines:
            if separate_propositions:
                line_propositions = line.split(";")
            else:
                line_propositions = [line]
            num_propositions_in_lines.append(len(line_propositions))
            for proposition in line_propositions:
                words = word_tokenize(proposition.lower())
                for word in words:
                    if word not in self.word_index:
                        self.word_index[word] = len(self.word_index)
                indices.append([self.word_index[word] for word in words])
        if pad:
            return num_propositions_in_lines, self.pad_indices(indices)
        else:
            return num_propositions_in_lines, indices

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
        indices_to_ignore = set([0])
        for token in [",", "(", ")", "."]:
            if token in self.word_index:
                indices_to_ignore.add(self.word_index[token])
        for indices in all_indices:
            corrupted_indices = list(indices)
            num_corrupted_locations = 0
            while num_corrupted_locations < num_locations_to_corrupt:
                rand_location = random.randint(0, len(indices)-1)
                if corrupted_indices[rand_location] in indices_to_ignore:
                    continue
                rand_index = 0
                while rand_index in indices_to_ignore:
                    rand_index = random.randint(1, len(self.word_index)-1)
                corrupted_indices[rand_location] = rand_index
                num_corrupted_locations += 1
            all_corrupted_indices.append(corrupted_indices)
        return all_corrupted_indices
