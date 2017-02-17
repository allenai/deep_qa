from collections import defaultdict


class DataIndexer:
    """
    A DataIndexer maps strings to integers, allowing for strings to be mapped to an
    out-of-vocabulary token.

    DataIndexers are fit to a particular dataset, which we use to decide which words are
    in-vocabulary.

    DataIndexers also allow for several different namespaces, so you can have separate word indices
    for 'a' as a word, and 'a' as a character, for instance.  Most of the methods on this class
    allow you to pass in a namespace; by default we use the 'words' namespace, and you can omit the
    namespace argument everywhere and just use the default.
    """
    def __init__(self):
        # Typically all input words to this code are lower-cased, so we could simply use "PADDING"
        # for this.  But doing it this way, with special characters, future-proofs the code in case
        # it is used later in a setting where not all input is lowercase.
        self._padding_token = "@@PADDING@@"
        self._oov_token = "@@UNKOWN@@"
        self.word_indices = defaultdict(lambda: {self._padding_token: 0, self._oov_token: 1})
        self.reverse_word_indices = defaultdict(lambda: {0: self._padding_token, 1: self._oov_token})

    def fit_word_dictionary(self, dataset: 'TextDataset', min_count: int=1):
        """
        Given a Dataset, this method decides which words are given an index, and which ones are
        mapped to an OOV token (in this case "UNK").  This method must be called before any dataset
        is indexed with this DataIndexer.  If you don't first fit the word dictionary, you'll
        basically map every token onto "UNK".

        We call instance.words() for each instance in the dataset, and then keep all words that
        appear at least min_count times.
        """
        namespace_word_counts = defaultdict(lambda: defaultdict(int))
        for instance in dataset.instances:
            namespace_dict = instance.words()
            for namespace in namespace_dict:
                for word in namespace_dict[namespace]:
                    namespace_word_counts[namespace][word] += 1
        for namespace in namespace_word_counts:
            for word, count in namespace_word_counts[namespace].items():
                if count >= min_count:
                    self.add_word_to_index(word, namespace)

    def add_word_to_index(self, word: str, namespace: str='words') -> int:
        """
        Adds `word` to the index, if it is not already present.  Either way, we return the index of
        the word.
        """
        if word not in self.word_indices[namespace]:
            index = len(self.word_indices[namespace])
            self.word_indices[namespace][word] = index
            self.reverse_word_indices[namespace][index] = word
            return index
        else:
            return self.word_indices[namespace][word]

    def words_in_index(self, namespace: str='words'):
        return self.word_indices[namespace].keys()

    def get_word_index(self, word: str, namespace: str='words'):
        if word in self.word_indices[namespace]:
            return self.word_indices[namespace][word]
        else:
            return self.word_indices[namespace][self._oov_token]

    def get_word_from_index(self, index: int, namespace: str='words'):
        return self.reverse_word_indices[namespace][index]

    def get_vocab_size(self, namespace: str='words'):
        return len(self.word_indices[namespace])
