from typing import Any, Dict, List, Tuple

from overrides import overrides

from .tokenizer import Tokenizer
from .word_processor import WordProcessor
from ..data_indexer import DataIndexer


class WordTokenizer(Tokenizer):
    """
    A ``WordTokenizer`` splits strings into word tokens.

    There are several ways that you can split a string into words, so we rely on a
    ``WordProcessor`` to do that work for us.  Note that we're using the word "tokenizer" here for
    something different than is typical in NLP - we're referring here to how strings are
    represented as numpy arrays, not the linguistic notion of splitting sentences into tokens.
    Those things are handled in the ``WordProcessor``, which is a common dependency in several
    ``Tokenizers``.

    Parameters
    ----------
    processor: Dict[str, Any], default={}
        Contains parameters for processing text strings into word tokens, including, e.g.,
        splitting, stemming, and filtering words.  See ``WordProcessor`` for a complete description
        of available parameters.
    """
    def __init__(self, params: Dict[str, Any]):
        self.word_processor = WordProcessor(params.pop('processor', {}))
        super(WordTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return self.word_processor.get_tokens(text)

    @overrides
    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        return {'words': self.tokenize(text)}

    @overrides
    def index_text(self, text: str, data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(word, namespace='words') for word in self.tokenize(text)]

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
        return {'num_sentence_words': sentence_length}
