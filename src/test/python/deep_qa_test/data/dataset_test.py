# pylint: disable=no-self-use
from pyfakefs import fake_filesystem_unittest

# These lines are just here so that `py.test --cov=deep_qa` produces a report for all deep_qa
# submodules.  It's not perfect (some files are still missing), but at least it's an improvement.
# These imports can be removed once there are actual tests for code in these modules.
import deep_qa.layers  # pylint: disable=unused-import
import deep_qa.sentence_corruption  # pylint: disable=unused-import
import deep_qa.solvers  # pylint: disable=unused-import

from deep_qa.data.dataset import Dataset, TextDataset
from deep_qa.data.text_instance import TrueFalseInstance

class TestDataset:
    def test_merge(self):
        instances = [TrueFalseInstance("testing", None, None), TrueFalseInstance("testing1", None, None)]
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
        dataset = TextDataset.read_from_file(filename, TrueFalseInstance)
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
