from collections import OrderedDict

from .entailment.snli_dataset import SnliDataset
from .language_modeling.language_modeling_dataset import LanguageModelingDataset
from .dataset import Dataset, TextDataset, IndexedDataset


concrete_datasets = OrderedDict()  # pylint: disable=invalid-name
concrete_datasets["text"] = TextDataset
concrete_datasets["language_modeling"] = LanguageModelingDataset
concrete_datasets["snli"] = SnliDataset
