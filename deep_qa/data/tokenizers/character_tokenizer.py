from typing import Callable, Dict, List, Tuple
from keras.layers import Layer
from overrides import overrides

from .tokenizer import Tokenizer
from ..data_indexer import DataIndexer
from ...common.params import Params


class CharacterTokenizer(Tokenizer):
    """
    A CharacterTokenizer splits strings into character tokens.
    """
    def __init__(self, params: Params):
        super(CharacterTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return list(text)

    @overrides
    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        return {'characters': self.tokenize(text)}

    @overrides
    def index_text(self,
                   text: str,
                   data_indexer: DataIndexer) -> List:
        return [data_indexer.get_word_index(char, namespace='characters') for char in self.tokenize(text)]

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    embed_function: Callable[[Layer, str, str], Layer],
                    text_trainer,
                    embedding_name: str="embedding"):
        return embed_function(input_layer,
                              embedding_name='character_' + embedding_name,
                              vocab_name='characters')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_padding_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        # Note that `sentence_length` here is the number of _characters_ in the sentence, because
        # of how `self.index_text` works.  And even though the name isn't great, we'll use
        # `num_sentence_words` for the key to this, so that the rest of the code is simpler.
        return {'num_sentence_words': sentence_length}
