from typing import Any, Dict, List, Tuple

from overrides import overrides

from .tokenizer import Tokenizer
from ..data_indexer import DataIndexer


class CharacterTokenizer(Tokenizer):
    """
    A CharacterTokenizer splits strings into character tokens.
    """
    def __init__(self, params: Dict[str, Any]):
        super(CharacterTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return [char for char in text]

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
                    input_layer: 'keras.layers.Layer',
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        # pylint: disable=protected-access
        return text_trainer._get_embedded_input(input_layer,
                                                embedding_name='character_' + embedding_name,
                                                vocab_name='characters')

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        return (sentence_length,)

    @overrides
    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        # Note that `sentence_length` here is the number of _characters_ in the sentence, because
        # of how `self.index_text` works.  And even though the name isn't great, we'll use
        # `num_sentence_words` for the key to this, so that the rest of the code is simpler.
        return {'num_sentence_words': sentence_length}
