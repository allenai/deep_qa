from typing import Any, Dict, List, Tuple

from ..data_indexer import DataIndexer
from ...common.params import ConfigurationError

class Tokenizer:
    """
    A Tokenizer splits strings into sequences of tokens that can be used in a model.  The "tokens"
    here could be words, characters, or words and characters.  The Tokenizer object handles various
    things involved with this conversion, including getting a list of tokens for pre-computing a
    vocabulary, getting the shape of a word sequence in a model, etc.  The Tokenizer needs to
    handle these things because the tokenization you do could affect the shape of word sequence
    tensors in the model (e.g., a sentence could have shape (num_words,), (num_characters,), or
    (num_words, num_characters)).
    """
    def __init__(self, params: Dict[str, Any]):
        # This class does not take any parameters, but for consistency in the API we take a params
        # dict as an argument.
        if len(params.keys()) != 0:
            raise ConfigurationError("You passed unrecognized parameters: " + str(params))

    def tokenize(self, text: str) -> List[str]:
        """
        Actually splits the string into a sequence of tokens.  Note that this will only give you
        top-level tokenization!  If you're using a word-and-character tokenizer, for instance, this
        will only return the word tokenization.
        """
        raise NotImplementedError

    def get_words_for_indexer(self, text: str) -> Dict[str, List[str]]:
        """
        The DataIndexer needs to assign indices to whatever strings we see in the training data
        (possibly doing some frequency filtering and using an OOV token).  This method takes some
        text and returns whatever the DataIndexer would be asked to index from that text.  Note
        that this returns a dictionary of token lists keyed by namespace.  Typically, the key would
        be either 'words' or 'characters'.  An example for indexing the string 'the' might be
        {'words': ['the'], 'characters': ['t', 'h', 'e']}, if you are indexing both words and
        characters.
        """
        raise NotImplementedError

    def index_text(self,
                   text: str,
                   data_indexer: DataIndexer) -> List:
        """
        This method actually converts some text into an indexed list.  This could be a list of
        integers (for either word tokens or characters), or it could be a list of arrays (for word
        tokens combined with characters), or something else.
        """
        raise NotImplementedError

    def embed_input(self,
                    input_layer: 'keras.layers.Layer',
                    text_trainer: 'TextTrainer',
                    embedding_name: str="embedding"):
        """
        Applies embedding layers to the input_layer.  See TextTrainer._embed_input for a more
        detailed comment on what this method does.

        - `input_layer` should be a Keras Input() layer.
        - `text_trainer` is a TextTrainer instance, so we can access methods on it like
          `text_trainer._get_embedded_input`, which actually applies an embedding layer, projection
          layers, and dropout to the input layer.  Simple TextEncoders will basically just call
          this function and be done.  More complicated TextEncoders might need additional logic on
          top of just calling `text_trainer._get_embedded_input`.
        - `embedding_name` allows for different embedding matrices.
        """
        raise NotImplementedError

    def get_sentence_shape(self, sentence_length: int, word_length: int) -> Tuple[int]:
        """
        If we have a text sequence of length `sentence_length`, what shape would that correspond to
        with this encoding?  For words or characters only, this would just be (sentence_length,).
        For an encoding that contains both words and characters, it might be (sentence_length,
        word_length).
        """
        raise NotImplementedError

    def get_max_lengths(self, sentence_length: int, word_length: int) -> Dict[str, int]:
        """
        When dealing with padding in TextTrainer, TextInstances need to know what to pad and how
        much.  This function takes a potential max sentence length and word length, and returns a
        `lengths` dictionary containing keys for the padding that is applicable to this encoding.
        """
        raise NotImplementedError

    def char_span_to_token_span(self,
                                sentence: str,
                                span: Tuple[int, int],
                                slack: int=3) -> Tuple[int, int]:
        """
        Converts a character span from a sentence into the corresponding token span in the
        tokenized version of the sentence.  If you pass in a character span that does not
        correspond to complete tokens in the tokenized version, we'll do our best, but the behavior
        is officially undefined.

        The basic outline of this method is to find the token that starts the same number of
        characters into the sentence as the given character span.  We try to handle a bit of error
        in the tokenization by checking `slack` tokens in either direction from that initial
        estimate.

        The returned ``(begin, end)`` indices are `inclusive` for ``begin``, and `exclusive` for
        ``end``.  So, for example, ``(2, 2)`` is an empty span, ``(2, 3)`` is the one-word span
        beginning at token index 2, and so on.
        """
        # First we'll tokenize the span and the sentence, so we can count tokens and check for
        # matches.
        span_chars = sentence[span[0]:span[1]]
        tokenized_span = self.tokenize(span_chars)
        tokenized_sentence = self.tokenize(sentence)
        # Then we'll find what we think is the first token in the span
        chars_seen = 0
        index = 0
        while index < len(tokenized_sentence) and chars_seen < span[0]:
            chars_seen += len(tokenized_sentence[index]) + 1
            index += 1
        # index is now the span start index.  Is it a match?
        if self._spans_match(tokenized_sentence, tokenized_span, index):
            return (index, index + len(tokenized_span))
        for i in range(1, slack + 1):
            if self._spans_match(tokenized_sentence, tokenized_span, index + i):
                return (index + i, index + i+ len(tokenized_span))
            if self._spans_match(tokenized_sentence, tokenized_span, index - i):
                return (index - i, index - i + len(tokenized_span))
        # No match; we'll just return our best guess.
        return (index, index + len(tokenized_span))

    @staticmethod
    def _spans_match(sentence_tokens: List[str], span_tokens: List[str], index: int) -> bool:
        if index < 0 or index >= len(sentence_tokens):
            return False
        if sentence_tokens[index] == span_tokens[0]:
            span_index = 1
            while (span_index < len(span_tokens) and
                   sentence_tokens[index + span_index] == span_tokens[span_index]):
                span_index += 1
            if span_index == len(span_tokens):
                return True
        return False
