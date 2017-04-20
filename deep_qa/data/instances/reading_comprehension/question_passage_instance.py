from typing import Dict, List, Any

import numpy as np
from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class QuestionPassageInstance(TextInstance):
    """
    A QuestionPassageInstance is a base class for datasets that consist primarily of a question
    text and a passage, where the passage contains the answer to the question. This class should
    not be used directly due to the missing ``_index_label`` function, use a subclass instead.
    """
    def __init__(self, question_text: str, passage_text: str, label: Any, index: int=None):
        super(QuestionPassageInstance, self).__init__(label, index)
        self.question_text = question_text
        self.passage_text = passage_text

    def __str__(self):
        return ('QuestionPassageInstance(' + self.question_text +
                ', ' + self.passage_text + ', ' +
                str(self.label) + ')')

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = self._words_from_text(self.question_text)
        passage_words = self._words_from_text(self.passage_text)
        for namespace in words:
            words[namespace].extend(passage_words[namespace])
        return words

    def _index_label(self, label: Any) -> List[int]:
        """
        Index the labels. Since we don't know what form the label takes,
        we leave it to subclasses to implement this method.
        """
        raise NotImplementedError

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        question_indices = self._index_text(self.question_text, data_indexer)
        passage_indices = self._index_text(self.passage_text, data_indexer)
        label_indices = self._index_label(self.label)
        return IndexedQuestionPassageInstance(question_indices,
                                              passage_indices, label_indices,
                                              self.index)

class IndexedQuestionPassageInstance(IndexedInstance):
    """
    This is an indexed instance that is used for (question, passage) pairs.
    """
    def __init__(self,
                 question_indices: List[int],
                 passage_indices: List[int],
                 label: List[int],
                 index: int=None):
        super(IndexedQuestionPassageInstance, self).__init__(label, index)
        self.question_indices = question_indices
        self.passage_indices = passage_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedQuestionPassageInstance([], [], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        We need to pad at least the question length, the passage length, and the
        word length across all the questions and passages. Subclasses that
        add more arguments should also override this method to enable padding on said
        arguments.
        """
        question_lengths = self._get_word_sequence_lengths(self.question_indices)
        passage_lengths = self._get_word_sequence_lengths(self.passage_indices)
        lengths = {}

        # the number of words to pad the question to
        lengths['num_question_words'] = question_lengths['num_sentence_words']

        # the number of words to pad the passage to
        lengths['num_passage_words'] = passage_lengths['num_sentence_words']

        if 'num_word_characters' in question_lengths and 'num_word_characters' in passage_lengths:
            # the length of the longest word across the passage and question
            lengths['num_word_characters'] = max(question_lengths['num_word_characters'],
                                                 passage_lengths['num_word_characters'])
        return lengths

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        In this function, we pad the questions and passages (in terms of number of words in each),
        as well as the individual words in the questions and passages themselves.
        """
        padding_lengths_tmp = padding_lengths.copy()
        padding_lengths_tmp['num_sentence_words'] = padding_lengths_tmp['num_question_words']
        self.question_indices = self.pad_word_sequence(self.question_indices, padding_lengths_tmp)
        padding_lengths_tmp['num_sentence_words'] = padding_lengths_tmp['num_passage_words']
        self.passage_indices = self.pad_word_sequence(self.passage_indices, padding_lengths_tmp,
                                                      truncate_from_right=False)

    @overrides
    def as_training_data(self):
        question_array = np.asarray(self.question_indices, dtype='int32')
        passage_array = np.asarray(self.passage_indices, dtype='int32')
        return (question_array, passage_array), np.asarray(self.label)
