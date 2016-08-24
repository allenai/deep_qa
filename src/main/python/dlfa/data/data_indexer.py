from collections import defaultdict


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
            if count > min_count:
                self.add_word_to_index(word)

    def add_word_to_index(self, word: str) -> int:
        """
        Adds `word` to the index, if it is not already present.  Either way, we return the index of
        the word.
        """
        if word not in self.word_index:
            index = len(self.word_index)
            self.word_index[word] = index
            self.reverse_word_index[index] = word
            return index
        else:
            return self.word_index[word]

    def words_in_index(self):
        return self.word_index.keys()

    def get_word_index(self, word: str):
        if word in self.word_index:
            return self.word_index[word]
        else:
            return self.word_index["UNK"]

    def get_word_from_index(self, index: int):
        return self.reverse_word_index[index]

    def get_vocab_size(self):
        return len(self.word_index) + 1
