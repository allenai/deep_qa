from collections import OrderedDict

from .word_overlap_tuple_match import WordOverlapTupleMatch

# The first item added here will be used as the default in some cases.
tuple_matchers = OrderedDict() # pylint: disable=invalid-name
tuple_matchers['word_overlap'] = WordOverlapTupleMatch
