from typing import Any, Dict, List, Tuple

from overrides import overrides

from .tokenizer import Tokenizer
from .word_splitter import word_splitters
from ..data_indexer import DataIndexer
from ...common.params import get_choice_with_default


class WordTokenizer(Tokenizer):
    """
    A WordTokenizer splits strings into word tokens.

    There are several ways that you can split a string into words, so we rely on a WordSplitter to
    do that work for us.  What we're calling a WordSplitter is typically called a "tokenizer" in
    NLP, but we're using WordSplitter here because for us "tokenization" is about whether you want
    words, characters, or both.
    """
    def __init__(self, params: Dict[str, Any]):
        word_splitter_choice = get_choice_with_default(params, 'word_splitter', list(word_splitters.keys()))
        self.word_splitter = word_splitters[word_splitter_choice]()
        super(WordTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return self.word_splitter.split_words(text)

    @overrides
    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        return {'words': self.word_splitter.split_words(text)}

    @overrides
    def index_text(self,
                   text: str,
                   data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(word, namespace='words')
                for word in self.word_splitter.split_words(text)]

    @overrides
    def embed_input(self,
                    input_layer: 'keras.layers.Layer',
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        # pylint: disable=protected-access
        return text_trainer._get_embedded_input(input_layer,
                                                embedding_name='word_' + embedding_name,
                                                vocab_name='words')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        return {'word_sequence_length': sentence_length}
