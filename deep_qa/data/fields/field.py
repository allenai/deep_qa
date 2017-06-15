"""
A ``Field`` is some piece of a data instance that ends up as an array in a model (either as an
input or an output).  Data instances are just collections of fields.


Fields go through up to two steps of processing: (1) tokenized fields are converted into token ids,
(2) fields containing token ids (or any other numeric data) are padded (if necessary) and converted
into data arrays.  We have two ``Field`` objects corresponding to fields in each stage of this
processing pipeline: ``UnindexedFields`` are fields containing raw strings (already tokenized) that
need to be converted into token IDs.  We use these fields to compute a vocabulary, then convert
them to ``IndexedFields``.  ``IndexedFields`` have a method for determining padding lengths, so
that a data generator object can intelligently batch together instances, then pad them.

If you are writing a new ``Field`` class, if there is nothing in the field that needs to be
converted from strings to integers, you should just subclass ``IndexedField``.  If you `do` need to
convert strings to integers, you should subclass both ``IndexedField`` and ``UnindexedField``, and
make the ``UnindexedField`` subclass public, and the other private.
"""
from typing import Dict, List

import numpy

from ..vocabulary import Vocabulary

# Because of how the dependencies work between these objects, we're defining them in reverse order,
# so that UnindexedField can reference IndexedField in its return types.

class IndexedField:
    """
    An ``IndexedField`` is a field that is almost ready to be put into a data array for use in a
    model.  The only thing missing is padding.  The methods on this object allow you to get padding
    lengths, so you can group things into batches of similar size, and then actually pad the
    field into a fixed-size array that can be used in a batch of instances.
    """
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        If there are things in this field that need padding, note them here.  In order to pad a
        batch of instance, we get all of the lengths from the batch, take the max, and pad
        everything to that length (or use a pre-specified maximum length).  The return value is a
        dictionary mapping keys to lengths, like {'num_tokens': 13}.
        """
        raise NotImplementedError

    def pad(self, padding_lengths: Dict[str, int]) -> List[numpy.array]:
        """
        Given a set of specified padding lengths, actually pad the data in this field and return a
        numpy array of the correct shape.  This actually returns a list instead of a single array,
        in case there are several related arrays for this field (e.g., a ``TextField`` might have a
        word array and a characters-per-word array).
        """
        raise NotImplementedError


class UnindexedField:
    """
    An ``UnindexedField`` is a field that still has strings instead of token ids.  We use these
    fields to compute vocabularies, and then convert them into ``IndexedFields``, from which we can
    actually construct data arrays.
    """
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        """
        If there are strings in this field that need to be converted into integers through a
        :class:`Vocabulary`, here is where we count them, to determine which tokens are in or out
        of the vocabulary.
        """
        raise NotImplementedError

    def index(self, vocab: Vocabulary) -> IndexedField:
        """
        Given a :class:`Vocabulary`, converts all tokens in this field into (typically) integer
        arrays, returning an ``IndexedField`` as a result.
        """
        raise NotImplementedError
