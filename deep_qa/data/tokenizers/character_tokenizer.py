from typing import Callable, Dict, List, Tuple
from keras.layers import Layer
from overrides import overrides

from .tokenizer import Tokenizer
from ..data_indexer import DataIndexer
from ...common.params import Params


class CharacterTokenizer(Tokenizer):
    """
    A CharacterTokenizer splits strings into character tokens.

    Notes
    -----
    Note that in the code, we're still using the "words" namespace, and the "num_sentence_words"
    padding key, instead of using a different "characters" namespace.  This is so that the rest of
    the code doesn't have to change as much to just use this different tokenizer.  For example,
    this is an issue when adding start and stop tokens - how is an ``Instance`` class supposed to
    know if it should use the "words" or the "characters" namespace when getting a start token id?
    If we just always use the "words" namespace for the top-level token namespace, it's not an
    issue.

    But confusingly, we'll still use the "characters" embedding key...  At least the user-facing
    parts all use ``characters``; it's only in writing tokenizer code that you need to be careful
    about namespaces.  TODO(matt): it probably makes sense to change the default namespace to
    "tokens", and use that for both the words in ``WordTokenizer`` and the characters in
    ``CharacterTokenizer``, so the naming isn't so confusing.
    """
    def __init__(self, params: Params):
        super(CharacterTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return list(text)

    @overrides
    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        return {'words': self.tokenize(text)}

    @overrides
    def index_text(self,
                   text: str,
                   data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(char) for char in self.tokenize(text)]

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    embed_function: Callable[[Layer, str, str], Layer],
                    text_trainer,
                    embedding_suffix: str=''):
        return embed_function(input_layer,
                              embedding_name='characters' + embedding_suffix,
                              vocab_name='words')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_padding_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        # Note that `sentence_length` here is the number of _characters_ in the sentence, because
        # of how `self.index_text` works.  And even though the name isn't great, we'll use
        # `num_sentence_words` for the key to this, so that the rest of the code is simpler.
        return {'num_sentence_words': sentence_length}
