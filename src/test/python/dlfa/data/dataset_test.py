from dlfa.data.dataset import Dataset
from dlfa.data.instance import TextInstance

class TestDataset:
    def test_merge(self):
        instances = [TextInstance("testing", None, None), TextInstance("testing1", None, None)]
        dataset1 = Dataset(instances[:1])
        dataset2 = Dataset(instances[1:])
        merged = dataset1.merge(dataset2)
        assert merged.instances == instances
