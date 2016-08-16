import sys
print(sys.path)
from dlfa.data.dataset import Dataset
from dlfa.data.instance import Instance

def test_merge():
    instances = [Instance("testing", None, None), Instance("testing1", None, None)]
    dataset1 = Dataset(instances[:1])
    dataset2 = Dataset(instances[1:])
    merged = dataset1.merge(dataset2)
    assert merged.instances == instances
