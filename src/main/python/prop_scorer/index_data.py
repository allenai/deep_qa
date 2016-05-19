import random
from nltk.tokenize import word_tokenize

class DataIndexer(object):
    def __init__(self):
        self.word_index = {"PADDING":0}

    def get_indices(self, lines, pad=True):
        indices = []
        for line in lines:
            words = word_tokenize(line.lower())
            for word in words:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
            indices.append([self.word_index[word] for word in words])
        if pad:
            return self.pad_indices(indices)
        else:
            return indices

    def pad_indices(self, indices, maxlen=None):
        if maxlen is None:
            maxlen = max([len(ind) for ind in indices])
        padded_indices = []
        for ind in indices:
            p_ind = [0]*maxlen
            ind_len = min(len(ind), maxlen)
            p_ind[-ind_len:] = ind[-ind_len:]
            padded_indices.append(p_ind)
        return padded_indices

    def corrupt_indices(self, indices, num_locs=1):
        c_inds = []
        inds_to_ignore = set([0])
        for tok in [",", "(", ")", "."]:
            if tok in self.word_index:
                inds_to_ignore.add(self.word_index[tok])
        for ind in indices:
            c_ind = list(ind)
            corr_locs = 0
            while corr_locs < num_locs:
                rand_loc = random.randint(0, len(ind)-1)
                if c_ind[rand_loc] in inds_to_ignore:
                    continue
                rand_i = 0
                while rand_i in inds_to_ignore:
                    rand_i = random.randint(1, len(self.word_index)-1)
                c_ind[rand_loc] = rand_i
                corr_locs += 1
            c_inds.append(c_ind)
        return c_inds
