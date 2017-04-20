from typing import List

from overrides import overrides

from .question_answer_instance import IndexedQuestionAnswerInstance, QuestionAnswerInstance
from ...data_indexer import DataIndexer


class BabiInstance(QuestionAnswerInstance):
    """
    This is a QuestionAnswerInstance that has consistent answer options and so does not include
    these answer options when outputting numpy arrays in `as_training_data()`.  The only place this
    makes sense, really, is in the bAbI dataset, with models that have a final softmax over a fixed
    answer space.
    """
    def __init__(self, question_text: str, answer_options: List[str], label: int, index: int=None):
        super(BabiInstance, self).__init__(question_text, answer_options, label, index)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        i = super(BabiInstance, self).to_indexed_instance(data_indexer)
        return IndexedBabiInstance(i.question_indices, i.option_indices, i.label, i.index)


class IndexedBabiInstance(IndexedQuestionAnswerInstance):
    def __init__(self,
                 question_indices: List[int],
                 option_indices: List[List[int]],
                 label: int,
                 index: int=None):
        super(IndexedBabiInstance, self).__init__(question_indices, option_indices, label, index)

    @overrides
    def as_training_data(self):
        inputs, label = super(IndexedBabiInstance, self).as_training_data()
        return inputs[0], label
