"""
A ``TextField`` represents a string of text, the kind that you might want to represent with
standard word vectors, or pass through an LSTM.
"""
from typing import Dict, List

from overrides import overrides
import numpy

from ..token_indexers import TokenIndexer, TokenType
from ..vocabulary import Vocabulary
from .field import UnindexedField, IndexedField



class _IndexedTextField(IndexedField):
    def __init__(self, token_arrays: List[List[TokenType]], token_indexers: List[TokenIndexer]):
        self._token_arrays = token_arrays
        self._token_indexers = token_indexers
        token_array_lengths = set(len(array) for array in token_arrays)
        assert len(token_array_lengths) == 1, "Token array lengths should all match, but were " +\
                token_array_lengths

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        lengths = []
        for indexer, array in zip(self._token_indexers, self._token_arrays):
            indexer_lengths = {}
            token_lengths = [indexer.get_padding_lengths(token) for token in array]
            for key in token_lengths[0].keys():
                indexer_lengths[key] = max(x[key] if key in x else 0 for x in token_lengths)
            lengths.append(indexer_lengths)
        padding_lengths = {'num_tokens': len(self._token_arrays[0])}
        for key in indexer_lengths[0].keys():
            padding_lengths[key] = max(x[key] if key in x else 0 for x in indexer_lengths)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        arrays = []
        desired_num_tokens = padding_lengths['num_tokens']
        for indexer, array in zip(self._token_indexers, self._token_arrays):
            padded_array = indexer.pad_token_sequence(array, desired_num_tokens, padding_lengths)
            arrays.append(padded_array)
        return arrays


class TokenizedTextField(UnindexedField):
    """
    This ``Field`` represents a list of string tokens.  Before constructing this object, you need
    to tokenize raw strings using a :class:`..tokenizers.Tokenizer`.

    Because string tokens can be represented as indexed arrays in a number of ways, we also take a
    list of :class:`TokenIndexer` objects that will be used to convert the tokens into indices.
    Each ``TokenIndexer`` could represent each token as a single ID, or a list of character IDs, or
    something else.
    """
    def __init__(self, tokens: List[str], token_indexers: List[TokenIndexer]):
        self._tokens = tokens
        self._token_indexers = token_indexers

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        for indexer in self._token_indexers:
            for token in self._tokens:
                indexer.count_vocab_items(token, counter)

    @overrides
    def index(self, vocab: Vocabulary) -> _IndexedTextField:
        token_arrays = []
        for indexer in self._token_indexers:
            token_arrays.append([indexer.token_to_indices(token, vocab) for token in self._tokens])
        return _IndexedTextField(token_arrays, self._token_indexers)
