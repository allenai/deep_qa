# pylint: disable=no-self-use,invalid-name
from deep_qa.data.datasets import SnliDataset
from deep_qa.data.instances.entailment.snli_instance import SnliInstance

from tests.common.test_case import DeepQaTestCase


class TestSnliDataset(DeepQaTestCase):

    def setUp(self):
        super(TestSnliDataset, self).setUp()
        self.write_original_snli_data()

    def test_read_from_file(self):
        dataset = SnliDataset.read_from_file(self.TRAIN_FILE, SnliInstance)

        instance1 = SnliInstance("A person on a horse jumps over a broken down airplane.",
                                 "A person is training his horse for a competition.",
                                 "neutral")
        instance2 = SnliInstance("A person on a horse jumps over a broken down airplane.",
                                 "A person is at a diner, ordering an omelette.",
                                 "contradicts")
        instance3 = SnliInstance("A person on a horse jumps over a broken down airplane.",
                                 "A person is outdoors, on a horse.",
                                 "entails")

        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.index == instance1.index
        assert instance.first_sentence == instance1.first_sentence
        assert instance.second_sentence == instance1.second_sentence
        assert instance.label == instance1.label
        instance = dataset.instances[1]
        assert instance.index == instance2.index
        assert instance.first_sentence == instance2.first_sentence
        assert instance.second_sentence == instance2.second_sentence
        assert instance.label == instance2.label
        instance = dataset.instances[2]
        assert instance.index == instance3.index
        assert instance.first_sentence == instance3.first_sentence
        assert instance.second_sentence == instance3.second_sentence
        assert instance.label == instance3.label
