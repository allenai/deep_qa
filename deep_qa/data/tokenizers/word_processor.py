from typing import List

from .word_splitter import word_splitters
from .word_stemmer import word_stemmers
from .word_filter import word_filters
from ...common.params import Params


class WordProcessor:
    """
    A WordProcessor handles the splitting of strings into words (with the use of a WordSplitter) as well as any
    desired post-processing (e.g., stemming, filtering, etc.)

    Parameters
    ----------
    word_splitter: str, default="simple"
        The string name of the ``WordSplitter`` of choice (see the options at the bottom of
        ``word_splitter.py``).

    word_filter: str, default="pass_through"
        The name of the ``WordFilter`` to use (see the options at the bottom of
        ``word_filter.py``).

    word_stemmer: str, default="pass_through"
        The name of the ``WordStemmer`` to use (see the options at the bottom of
        ``word_stemmer.py``).
    """
    def __init__(self, params: Params):
        word_splitter_choice = params.pop_choice('word_splitter', list(word_splitters.keys()),
                                                 default_to_first_choice=True)
        self.word_splitter = word_splitters[word_splitter_choice]()
        word_filter_choice = params.pop_choice('word_filter', list(word_filters.keys()),
                                               default_to_first_choice=True)
        self.word_filter = word_filters[word_filter_choice]()
        word_stemmer_choice = params.pop_choice('word_stemmer', list(word_stemmers.keys()),
                                                default_to_first_choice=True)
        self.word_stemmer = word_stemmers[word_stemmer_choice]()
        params.assert_empty("WordProcessor")

    def get_tokens(self, sentence: str) -> List[str]:
        """
        Does whatever processing is required to convert a string of text into a sequence of tokens.

        At a minimum, this uses a ``WordSplitter`` to split words into text.  It may also do
        stemming or stopword removal, depending on the parameters given to the constructor.
        """
        words = self.word_splitter.split_words(sentence)
        filtered_words = self.word_filter.filter_words(words)
        stemmed_words = [self.word_stemmer.stem_word(word) for word in filtered_words]
        return stemmed_words
