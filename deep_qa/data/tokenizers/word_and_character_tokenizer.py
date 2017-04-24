from typing import Any, Callable, Dict, List, Tuple

from overrides import overrides
from keras import backend as K
from keras.layers import Concatenate, Layer

from .tokenizer import Tokenizer
from .word_processor import WordProcessor
from ..data_indexer import DataIndexer
from ...layers.backend import CollapseToBatch
from ...layers.backend import ExpandFromBatch
from ...layers.wrappers import EncoderWrapper
from ...layers import VectorMatrixSplit
from ...common.params import Params


class WordAndCharacterTokenizer(Tokenizer):
    """
    A ``WordAndCharacterTokenizer`` first splits strings into words, then splits those words into
    characters, and returns a representation that contains `both` a word index and a sequence of
    character indices for each word.  See the documention for ``WordTokenizer`` for a note about
    naming, and the typical notion of "tokenization" in NLP.

    Notes
    -----
    In ``embed_input``, this ``Tokenizer`` uses an encoder to get a character-level word embedding,
    which then gets concatenated with a standard word embedding from an embedding matrix.  To
    specify the encoder to use for this character-level word embedding, use the ``"word"`` key in
    the ``encoder`` parameter to your model (which should be a ``TextTrainer`` subclass - see the
    documentation there for some more info).  If you do not give a ``"word"`` key in the
    ``encoder`` dict, we'll create a new encoder using the ``"default"`` parameters.
    """
    def __init__(self, params: Params):
        self.word_processor = WordProcessor(params.pop('processor', {}))
        super(WordAndCharacterTokenizer, self).__init__(params)

    @overrides
    def tokenize(self, text: str) -> List[str]:
        return self.word_processor.get_tokens(text)

    @overrides
    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        words = self.tokenize(text)
        characters = [char for word in words for char in word]
        return {'words': words, 'characters': characters}

    @overrides
    def index_text(self, text: str, data_indexer: DataIndexer) -> List:
        words = self.tokenize(text)
        arrays = []
        for word in words:
            word_index = data_indexer.get_word_index(word, namespace='words')
            # TODO(matt): I'd be nice to keep the capitalization of the word in the character
            # representation.  Doing that would require pretty fancy logic here, though.
            char_indices = [data_indexer.get_word_index(char, namespace='characters') for char in word]
            arrays.append([word_index] + char_indices)
        return arrays

    @overrides
    def embed_input(self,
                    input_layer: Layer,
                    embed_function: Callable[[Layer, str, str], Layer],
                    text_trainer,
                    embedding_name: str="embedding"):
        """
        A combined word-and-characters representation requires some fancy footwork to do the
        embedding properly.

        This method assumes the input shape is (..., sentence_length, word_length + 1), where the
        first integer for each word in the tensor is the word index, and the remaining word_length
        entries is the character sequence.  We'll first split this into two tensors, one of shape
        (..., sentence_length), and one of shape (..., sentence_length, word_length), where the
        first is the word sequence, and the second is the character sequence for each word.  We'll
        pass the word sequence through an embedding layer, as normal, and pass the character
        sequence through a _separate_ embedding layer, then an encoder, to get a word vector out.
        We'll then concatenate the two word vectors, returning a tensor of shape
        (..., sentence_length, embedding_dim * 2).
        """
        # pylint: disable=protected-access
        # This is happening before any masking is done, so we don't need to worry about the
        # mask_split_axis argument to VectorMatrixSplit.
        words, characters = VectorMatrixSplit(split_axis=-1)(input_layer)
        word_embedding = embed_function(words,
                                        embedding_name='word_' + embedding_name,
                                        vocab_name='words')
        character_embedding = embed_function(characters,
                                             embedding_name='character_' + embedding_name,
                                             vocab_name='characters')

        # A note about masking here: we care about the character masks when encoding a character
        # sequence, so we need the mask to be passed to the character encoder correctly.  However,
        # we _don't_ care here about whether the whole word will be masked, as the word_embedding
        # will carry that information.  Because of the way `Concatenate` handles masks, if you've
        # done something crazy where you have a word index but your character indices are all zero,
        # you will get a 0 in the mask for that word at the end of this.  But assuming you're using
        # this correctly, you should only get a 0 in the character-level mask in the same places
        # that you have 0s in the word-level mask, so `Concatenate` further below will do the right
        # thing.

        # character_embedding has shape `(batch_size, ..., num_words, word_length,
        # embedding_dim)`, but our encoder expects a tensor with shape `(batch_size,
        # word_length, embedding_dim)` to the word encoder.  Typically, we would use Keras'
        # TimeDistributed layer to handle this.  However, if we're using dynamic padding, we
        # may not know `num_words` (which we're collapsing) or even `word_length` at runtime,
        # which messes with TimeDistributed.  In order to handle this correctly, we'll use
        # CollapseToBatch and ExpandFromBatch instead of TimeDistributed.  Those layers
        # together do basically the same thing, collapsing all of the unwanted dimensions into
        # the batch_size temporarily, but they can handle unknown runtime shapes.
        dims_to_collapse = K.ndim(character_embedding) - 3
        collapsed_character_embedding = CollapseToBatch(dims_to_collapse)(character_embedding)
        word_encoder = text_trainer._get_encoder(name="word", fallback_behavior="use default params")
        collapsed_word_encoding = word_encoder(collapsed_character_embedding)
        word_encoding = ExpandFromBatch(dims_to_collapse)([collapsed_word_encoding, character_embedding])
        # If you're embedding multiple inputs in your model, we need the final concatenation here
        # to have a unique name each time.  In order to get a unique name, we use the name of the
        # input layer.  Except sometimes Keras adds funny things to the ends of the input layer, so
        # we'll strip those off.
        input_name = input_layer.name
        if '/' in input_name:
            input_name = input_name.rsplit('/', 1)[1]
        if ':' in input_name:
            input_name = input_name.split(':')[0]
        if input_name.split('_')[-1].isdigit():
            input_name = '_'.join(input_name.split('_')[:-1])
        name = 'combined_word_embedding_for_' + input_name
        final_embedded_input = Concatenate(name=name)([word_embedding, word_encoding])
        return final_embedded_input

    @overrides
    def get_sentence_shape(self, sentence_length: int, word_length: int=None) -> Tuple[int]:
        return (sentence_length, word_length)

    @overrides
    def get_padding_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        return {'num_sentence_words': sentence_length, 'num_word_characters': word_length}

    @overrides
    def get_custom_objects(self) -> Dict[str, Any]:
        return {
                'CollapseToBatch': CollapseToBatch,
                'EncoderWrapper': EncoderWrapper,
                'ExpandFromBatch': ExpandFromBatch,
                'VectorMatrixSplit': VectorMatrixSplit,
                }
