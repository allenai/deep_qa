from typing import Tuple, List

import numpy
from overrides import overrides

from .question_passage_instance import QuestionPassageInstance, IndexedQuestionPassageInstance
from ..data_indexer import DataIndexer


class CharacterSpanInstance(QuestionPassageInstance):
    """
    A CharacterSpanInstance is a QuestionPassageInstance that represents a (question, passage) pair
    with an associated label, which is the data given for the span prediction task. The label is a
    span of characters in the passage that indicates where the answer to the question begins and
    where the answer to the question ends.

    The main thing this class handles over QuestionPassageInstance is in specifying the form of and
    how to index the label, which is given as a span of _characters_ in the passage. The label we
    are going to use in the rest of the code is a span of _tokens_ in the passage, so the mapping
    from character labels to token labels depends on the tokenization we did, and the logic to
    handle this is, unfortunately, a little complicated. The label conversion happens when
    converting a CharacterSpanInstance to in IndexedInstance (where character indices are generally
    lost, anyway).

    This class should be used to represent training instances for the SQuAD (Stanford Question
    Answering) and NewsQA datasets, to name a few.
    """
    # We add a special token to the end of the passage.  This is because our span labels are
    # end-exclusive, and we do a softmax over the passage to determine span end.  So if we want to
    # be able to include the last token of the passage, we need to have a special symbol at the
    # end.
    stop_token = "@@STOP@@"

    def __init__(self, question: str, passage: str, label: Tuple[int, int], index: int=None):
        super(CharacterSpanInstance, self).__init__(question, passage, label, index)

    def __str__(self):
        return ('CharacterSpanInstance(' + self.question_text + ', ' +
                self.passage_text + ', ' + str(self.label) + ')')

    @overrides
    def _index_label(self, label: Tuple[int, int]) -> List[int]:
        """
        Specify how to index `self.label`, which is needed to convert the CharacterSpanInstance
        into an IndexedInstance (handled in superclass).
        """
        if self.label is not None:
            return self.tokenizer.char_span_to_token_span(self.passage_text, self.label)
        return None

    @classmethod
    def read_from_line(cls, line: str, default_label: bool=None):
        """
        Reads a CharacterSpanInstance object from a line. The format has one of two options:

        (1) [example index][tab][question][tab][passage][tab][label]
        (2) [question][tab][passage][tab][label]

        [label] is assumed to be a comma-separated pair of integers.

        default_label is ignored, but we keep the argument to match the interface.
        """
        fields = line.split("\t")

        if len(fields) == 4:
            index_string, question, passage, label = fields
            index = int(index_string)
        elif len(fields) == 3:
            question, passage, label = fields
            index = None
        else:
            raise RuntimeError("Unrecognized line format (" + str(len(fields)) + " columns): " + line)
        label_fields = label.split(",")
        span_begin = int(label_fields[0])
        span_end = int(label_fields[1])
        return cls(question, passage, (span_begin, span_end), index)

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        instance = super(CharacterSpanInstance, self).to_indexed_instance(data_indexer)
        stop_index = data_indexer.add_word_to_index(self.stop_token)
        if isinstance(instance.passage_indices[0], list):
            instance.passage_indices.append([stop_index])
        else:
            instance.passage_indices.append(stop_index)
        return IndexedCharacterSpanInstance(instance.question_indices, instance.passage_indices,
                                            instance.label, instance.index)


class IndexedCharacterSpanInstance(IndexedQuestionPassageInstance):
    @overrides
    def as_training_data(self):
        input_arrays, _ = super(IndexedCharacterSpanInstance, self).as_training_data()
        span_begin_label = span_end_label = None
        if self.label is not None:
            span_begin_label = numpy.zeros((len(self.passage_indices)))
            span_end_label = numpy.zeros((len(self.passage_indices)))
            span_begin_label[self.label[0]] = 1
            span_end_label[self.label[1]] = 1
        return input_arrays, (span_begin_label, span_end_label)
