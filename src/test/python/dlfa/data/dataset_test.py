import os
import unittest
from pyfakefs import fake_filesystem_unittest

from dlfa.data.dataset import Dataset, TextDataset
from dlfa.data.instance import TextInstance

class TestDataset:
    def test_merge(self):
        instances = [TextInstance("testing", None, None), TextInstance("testing1", None, None)]
        dataset1 = Dataset(instances[:1])
        dataset2 = Dataset(instances[1:])
        merged = dataset1.merge(dataset2)
        assert merged.instances == instances

class TestTextDataset(fake_filesystem_unittest.TestCase):
    def setUp(self):
        self.setUpPyfakefs()

    def test_read_from_file_with_no_default_label(self):
        filename = '/test_dataset_file'
        with open(filename, 'w') as datafile:
            datafile.write("1\tinstance1\t0\n")
            datafile.write("2\tinstance2\t1\n")
            datafile.write("3\tinstance3\n")
        dataset = TextDataset.read_from_file(filename)
        assert len(dataset.instances) == 3
        instance = dataset.instances[0]
        assert instance.index == 1
        assert instance.text == "instance1"
        assert instance.label == False
        instance = dataset.instances[1]
        assert instance.index == 2
        assert instance.text == "instance2"
        assert instance.label == True
        instance = dataset.instances[2]
        assert instance.index == 3
        assert instance.text == "instance3"
        assert instance.label == None
