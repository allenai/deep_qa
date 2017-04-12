# pylint: disable=no-self-use,invalid-name
from deep_qa.common import util
from .test_case import DeepQaTestCase


class TestCommonUtils(DeepQaTestCase):
    def test_group_by_count(self):
        assert util.group_by_count([1, 2, 3, 4, 5, 6, 7], 3, 20) == [[1, 2, 3], [4, 5, 6], [7, 20, 20]]
