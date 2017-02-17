from collections import OrderedDict

from .character_tokenizer import CharacterTokenizer
from .word_and_character_tokenizer import WordAndCharacterTokenizer
from .word_tokenizer import WordTokenizer

# The first item added here will be used as the default in some cases.
tokenizers = OrderedDict()  # pylint: disable=invalid-name
tokenizers['words'] = WordTokenizer
tokenizers['characters'] = CharacterTokenizer
tokenizers['words and characters'] = WordAndCharacterTokenizer
