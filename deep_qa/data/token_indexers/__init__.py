from collections import OrderedDict

from .single_id_token_indexer import SingleIdTokenIndexer
from .token_characters_indexer import TokenCharactersIndexer
from .token_indexer import TokenIndexer, TokenType

# The first item added here will be used as the default in some cases.
token_indexers = OrderedDict()  # pylint: disable=invalid-name
token_indexers['single id'] = SingleIdTokenIndexer
token_indexers['characters'] = TokenCharactersIndexer
