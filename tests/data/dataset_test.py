# pylint: disable=no-self-use,invalid-name
from deep_qa.data.dataset import Dataset, TextDataset
from deep_qa.data.instances.text_classification.text_classification_instance import TextClassificationInstance
from ..common.test_case import DeepQaTestCase


class TestDataset:
    def test_merge(self):
        instances = [TextClassificationInstance("testing", None, None),
                     TextClassificationInstance("testing1", None, None)]
        dataset1 = Dataset(instances[:1])
        dataset2 = Dataset(instances[1:])
        merged = dataset1.merge(dataset2)
        assert merged.instances == instances


class TestTextDataset(DeepQaTestCase):
    def test_read_from_file_with_no_default_label(self):
        filename = self.TEST_DIR + 'test_dataset_file'
        with open(filename, 'w') as datafile:
            datafile.write("1\tinstance1\t0\n")
            datafile.write("2\tinstance2\t1\n")
            datafile.write("3\tinstance3\n")
        dataset = TextDataset.read_from_file(filename, TextClassificationInstance)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.index == 1
        assert instance.text == "instance1"
        assert instance.label is False
        instance = dataset.instances[1]
        assert instance.index == 2
        assert instance.text == "instance2"
        assert instance.label is True
        instance = dataset.instances[2]
        assert instance.index == 3
        assert instance.text == "instance3"
        assert instance.label is None
