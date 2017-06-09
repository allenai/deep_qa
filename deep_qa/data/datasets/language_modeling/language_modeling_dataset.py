from typing import List

from overrides import overrides

from ..dataset import TextDataset, log_label_counts
from ...instances import TextInstance
from ...instances.language_modeling import SentenceInstance
from ....common.params import Params


class LanguageModelingDataset(TextDataset):

    def __init__(self, instances: List[TextInstance], params: Params=None):
        # TODO(Mark): We are splitting on spaces below, so this won't end up being
        # the exact sequence length. This could be solved by passing the tokeniser
        # to the dataset.
        self.sequence_length = params.pop("sequence_length")
        super(LanguageModelingDataset, self).__init__(instances)

    @staticmethod
    @overrides
    def read_from_file(filename: str, instance_class, params: Params=None):

        sequence_length = params.get("sequence_length", 20)
        with open(filename, "r") as text_file:
            text = text_file.readlines()
            text = " ".join([x.replace("\n", " ").strip() for x in text]).split(" ")

        instances = []
        for index in range(0, len(text) - sequence_length, sequence_length):
            word_sequence = " ".join(text[index: index + sequence_length])
            instances.append(SentenceInstance(word_sequence))

        log_label_counts(instances)
        return LanguageModelingDataset(instances, params)
