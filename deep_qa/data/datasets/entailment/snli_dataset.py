from typing import List
import json

from overrides import overrides

from ..dataset import TextDataset, log_label_counts
from ...instances import TextInstance
from ....common.params import Params


class SnliDataset(TextDataset):

    def __init__(self, instances: List[TextInstance], params: Params=None):
        super(SnliDataset, self).__init__(instances, params)

    @staticmethod
    @overrides
    def read_from_file(filename: str, instance_class, params: Params=None):

        instances = []
        for line in open(filename, 'r'):
            example = json.loads(line)

            # TODO(mark) why does this not match snli? Fix.
            label = example["gold_label"]
            if label == "entailment":
                label = "entails"
            elif label == "contradiction":
                label = "contradicts"

            text = example["sentence1"]
            hypothesis = example["sentence2"]
            instances.append(instance_class(text, hypothesis, label))
        log_label_counts(instances)
        return SnliDataset(instances, params)
