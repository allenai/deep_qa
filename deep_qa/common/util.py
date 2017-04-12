from itertools import zip_longest
from typing import Any, List


def group_by_count(iterable: List[Any], count: int, default_value: Any) -> List[List[Any]]:
    """
    Takes a list and groups it into sublists of size ``count``, using ``default_value`` to pad the
    list at the end if the list is not divisable by ``count``.

    For example:
    >>> group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 0)
    [[1, 2, 3], [4, 5, 6], [7, 0, 0]]

    This is a short method, but it's complicated and hard to remember as a one-liner, so we just
    make a function out of it.
    """
    return [list(l) for l in zip_longest(*[iter(iterable)] * count, fillvalue=default_value)]
