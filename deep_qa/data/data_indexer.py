from collections import defaultdict
import codecs
import logging

import tqdm

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


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
        self._finalized = False

    def set_from_file(self, filename: str, oov_token: str="@@UNKNOWN@@", namespace: str="words"):
        self._oov_token = oov_token
        with codecs.open(filename, 'r', 'utf-8') as input_file:
            for line in input_file:
                self.add_word_to_index(line.rstrip(), namespace)

    def finalize(self):
        logger.info("Finalizing data indexer")
        self._finalized = True

    def fit_word_dictionary(self, dataset, min_count: int=1):
        """
        Given a ``Dataset``, this method decides which words are given an index, and which ones are
        mapped to an OOV token (in this case "UNK").  This method must be called before any dataset
        is indexed with this ``DataIndexer``.  If you don't first fit the word dictionary, you'll
        basically map every token onto "UNK".

        We call ``instance.words()`` for each instance in the dataset, and then keep all words that
        appear at least ``min_count`` times.

        Parameters
        ----------
        dataset: ``TextDataset``
            The dataset to index.

        min_count: int, optional (default=1)
            The minimum number of occurences a word must have in the dataset
            in order to be assigned an index.
        """
        logger.info("Fitting word dictionary with min count of %d, finalized is %s",
                    min_count, self._finalized)
        if self._finalized:
            logger.warning("Trying to fit a finalized DataIndexer.  This is a no-op.  Did you "
                           "really want to do this?")
            return
        namespace_word_counts = defaultdict(lambda: defaultdict(int))
        for instance in tqdm.tqdm(dataset.instances):
            namespace_dict = instance.words()
            for namespace in namespace_dict:
                for word in namespace_dict[namespace]:
                    namespace_word_counts[namespace][word] += 1
        for namespace in tqdm.tqdm(namespace_word_counts):
            for word, count in namespace_word_counts[namespace].items():
                if count >= min_count:
                    self.add_word_to_index(word, namespace)

    def fit_word_existing_dictionary(self, dataset, embeddings, use_rand = False):
        """
        Given a ``Dataset``, an embedding space and an existing index size,
        this method traverses the words in the datasets, and adds all words into the index.

        Parameters
        ----------
        dataset: ``TextDataset``
            The dataset to index.
        """
        logger.info("Fitting word dictionary with")

        namespace_word_counts = defaultdict(lambda: defaultdict(int))
        for instance in tqdm.tqdm(dataset.instances):
            namespace_dict = instance.words()
            for namespace in namespace_dict:
                # print(namespace_dict[namespace])
                for word in namespace_dict[namespace]:
                    namespace_word_counts[namespace][word] += 1

        for namespace in namespace_dict:
            logger.info("%s: Read %d words. Index size is %s",namespace, len(namespace_word_counts[namespace]), len(self.word_indices[namespace]))
            # for word in self.word_indices[namespace]:
            #     print("Orig:",word)

        n = 0
        n2 = 0
        nt = 0
        nt2 = 0
        nta = 0
        for namespace in tqdm.tqdm(namespace_word_counts):
            # Starting from 2, first two tokens are padding and oov
            i = 2
            for word, count in sorted(namespace_word_counts[namespace].items(), key=lambda x: x[1]):

                if word not in self.word_indices[namespace]:
                    n += 1
                    nt += namespace_word_counts[namespace][word]
                    # logger.info("New word: %s (%d)", word, namespace_word_counts[namespace][word])

                if word not in embeddings:
                    n2 += 1
                    nt2 += namespace_word_counts[namespace][word]
                    # try:
                    #     # print(word,namespace_word_counts[namespace][word],"is out!!!")
                    # except UnicodeEncodeError:
                        # print("##BadUnicode##", namespace_word_counts[namespace][word], "is out!!!")
                # print(word)

                nta += namespace_word_counts[namespace][word]

                if word in embeddings or use_rand:
                    # print("\t",word)
                    index = None
                    old_word = self.reverse_word_indices[namespace][i]
                    # If word is already in index, remove it
                    if word in self.word_indices[namespace]:
                        index = self.word_indices[namespace][word]

                        self.reverse_word_indices[namespace][index] = None

                    if old_word is not None:
                        del self.word_indices[namespace][old_word]

                    # print("\t\tBTATA", i, index, old_word, word)

                    self.word_indices[namespace][word] = i
                    self.reverse_word_indices[namespace][i] = word
                    i += 1

                    if i == len(self.word_indices[namespace]):
                        break

            orig_dict_size = len(self.reverse_word_indices[namespace])
            for j in range(i,orig_dict_size):
                old_word = self.reverse_word_indices[namespace][j]
                if old_word is not None:
                    del self.word_indices[namespace][old_word]
                self.reverse_word_indices[namespace][j] = None

            logger.info("N training words: %d %d. N test words: %d (%d)"". N new: %d. N not in embeddings: %d. Nta:"\
                        "%d,nt: %d, nt2: %d",orig_dict_size,len(self.word_indices[namespace]),len(namespace_word_counts[namespace]),i, n, n2, nta,nt, nt2)



    def add_word_to_index(self, word: str, namespace: str='words') -> int:
        """
        Adds `word` to the index, if it is not already present.  Either way, we return the index of
        the word.
        """
        if self._finalized:
            logger.warning("Trying to add a word to a finalized DataIndexer.  This is a no-op.  "
                           "Did you really want to do this?")
            return self.word_indices[namespace].get(word, -1)
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
