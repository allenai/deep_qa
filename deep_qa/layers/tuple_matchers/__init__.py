"""
``TupleMatchers`` take two tuples, which are passed in as a list of two tensor inputs, each of
shape (batch size, number tuple slots, number of tuple elements), and compare them to determine how
well they match.  The input is assumed to be `word indices`, not embedded representations.  For
example, an answer candidate tuple might be compared against a tuple that encodes background
knowledge to determine how well the background knowledge `entails` the answer candidate.

While typically the tuples will be the same shape, it need not necessarily be the case.  For
example, conceivably the two tuples might in the future have differing numbers of words in each
slot if different tuple creation methods are applied to different text sources.  Alternatively, if
broader context is used for tuples encoding background knowledge, one of the tuples could
potentially have additional slots.

``TupleMatchers`` should look like ``Layers`` to model code, though they need not actually `be`
Keras ``Layers``.  That is, you either need to inherit from ``Layer`` directly, or you need to
implement a ``__call__`` method that returns the output of applying a ``Layer`` to some input.  You
need to implement a ``TupleMatcher`` with a ``__call__`` method any time you want to actually apply
several ``Layers`` to the input (e.g., to embed the words before doing some computation on them).
"""
from collections import OrderedDict

from .slot_similarity_tuple_matcher import SlotSimilarityTupleMatcher
from .word_overlap_tuple_matcher import WordOverlapTupleMatcher
from .encoded_tuple_matcher import EncodedTupleMatcher
from .embedded_tuple_matcher import EmbeddedTupleMatcher
from .threshold_tuple_matcher import ThresholdTupleMatcher

# The first item added here will be used as the default in some cases.
tuple_matchers = OrderedDict() # pylint: disable=invalid-name
tuple_matchers['word_overlap'] = WordOverlapTupleMatcher
tuple_matchers['encoded_matcher'] = EncodedTupleMatcher
tuple_matchers['embedded_matcher'] = EmbeddedTupleMatcher
