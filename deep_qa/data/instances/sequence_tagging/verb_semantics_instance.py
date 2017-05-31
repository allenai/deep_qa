from typing import Dict, List, Tuple

import numpy

from overrides import overrides

from ..instance import TextInstance, IndexedInstance
from ...data_indexer import DataIndexer


class VerbSemanticsInstance(TextInstance):
    """
    A VerbSemanticsInstance is a :class:`TextInstance` that is a single sentence of text,
    along with query = (verb, entity) spans in the sentence
    Each of these instances have a label = (state_change_type, arg1 , arg2 )
    E.g. for the sentence = "roots absorb water from the soil"
         Query: verb = absorb,  entity = water
         Label: state_change= MOVE, arg1(location-from) = the soil , arg2 (location-to) = roots
    """
    def __init__(self, sentence: List[str], verb: (int, int), entity: (int, int),
                 state_change: str, arg1: (int, int), arg2: (int, int), index: int=None):
        """
        sentence: List[str]
            A pre-tokenized sentence from which we want to predict verb semantics.
            A sentence can mention multiple verbs and entities,
            a data instance is talking about a particular (verb, entity) pair in a given sentence.
        verb: (int, int)
            Span of the verb indicating the event being referred to in the query.
        entity: (int, int)
            Span of the entity for which we want to track the state change.
        state_change: str
            State change type.
        arg1: (int, int)
            Span of the argument1 of the state change.
        arg2: (int, int)
            Span of the argument2 of the state change.
        E.g. For the same example above,
            sentence = ["roots", "absorb", "water", "from", "the", "soil"]
            verb = (1,1)   // absorb
            entity = (2,2) // water
            arg1 = (4,5)   // the soil
            arg2 = (0,0)   // roots
        """
        self.sentence = sentence
        self.verb = verb
        self.entity = entity
        tags = self.__make_tag_sequence(arg1, arg2)
        super(VerbSemanticsInstance, self).__init__((state_change, tags), index)

    def __str__(self):
        return 'VerbSemanticsInstance(' + self.sentence + ', ' + self.verb + ', ' + \
               self.entity + str(self.label) + ')'

    def __make_tag_sequence(self, arg1: (int, int), arg2: (int, int)) -> List[str]:
        """
        Converts (arg1, arg2) into a tag sequence over the sentence.
        arg1: (int, int)
            StartIndex,endIndex of arg1 in the sentence.
        arg2: (int, int)
            StartIndex,endIndex of arg2 in the sentence.
        tag sequence: List[str]
            List of tag names one per token in the sentence.
        E.g. sentence = "Roots absorb water from the soil"
            arg1 = (4, 5)  // the soil
            arg2 = (0, 0)  // roots
            tag sequence = (B-ARG2, O, O, O, B-ARG1, I-ARG1)
        """
        tags = ["O"] * len(self.sentence)
        if arg1[0] != -1:
            tags[arg1[0]] = "B-" + "ARG1"
            for index1 in range(arg1[0]+1, arg1[1]):
                tags[index1] = "I-" + "ARG1"
        if arg2[0] != -1:
            tags[arg2[0]] = "B-" + "ARG2"
            for index2 in range(arg2[0]+1, arg2[1]):
                tags[index2] = "I-" + "ARG2"
        return tags

    @overrides
    def words(self) -> Dict[str, List[str]]:
        """
        Creates 3 different word dictionaries:
        1) words in the sentence
        2) state change types
        3) tags for arguments
        """
        words = {'words': self.sentence, 'state_changes': [self.label[0]], 'tags': self.label[1]}
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):

        sentence_indices = [data_indexer.get_word_index(word) for word in self.sentence]

        # verb indices, entity indices are one-hot vectors representing spans within the sentence
        verb_indices = [0] * len(sentence_indices)
        for entity_id in range(self.verb[0], self.verb[1] + 1):
            verb_indices[entity_id] = 1

        entity_indices = [0] * len(sentence_indices)
        for entity_id in range(self.entity[0], self.entity[1] + 1):
            entity_indices[entity_id] = 1

        # indexed_label is a tuple with one-hot vectors for state change and tags
        # state_change is a 1-hot vector of size #change_types
        # tags is a list of size sentence_length, where each element is a 1-hot vector of size #tag_types

        state_index = data_indexer.get_word_index(self.label[0], namespace='state_changes')
        state_one_hot = [0] * (data_indexer.get_vocab_size(namespace='state_changes') - 2)
        if state_index >= 2:
            state_one_hot[state_index - 2] = 1

        tag_one_hot_list = []
        tag_indices = [data_indexer.get_word_index(tag, namespace='tags') for tag in self.label[1]]
        for tag_index in tag_indices:
            # We subtract 2 here to account for the unknown and padding tokens that the DataIndexer uses.
            tag_one_hot = [0] * (data_indexer.get_vocab_size(namespace='tags') - 2)
            if tag_index >= 2:
                tag_one_hot[tag_index - 2] = 1
            tag_one_hot_list.append(tag_one_hot)

        indexed_label = state_one_hot, tag_one_hot_list

        return IndexedVerbSemanticsInstance(sentence_indices, verb_indices,
                                            entity_indices, indexed_label, self.index)

    @classmethod
    @overrides
    def read_from_line(cls, line: str):
        """
        Reads a VerbSemanticsInstance object from a line.  The format is as follows:
        [sentence][tab][verb][tab][entity][tab][stateChangeLabel][tab][arg1][tab][arg2]

        sentence: This is pre-tokenized with tokens separated by "####".
        verb: StartIndex,endIndex of verb in the sentence.
        entity: StartIndex,endIndex of entity in the sentence.
        state_change_label: One label from {CREATE, DESTROY, MOVE}.
        arg1: StartIndex,endIndex of arg1 in the sentence.
        arg2: StartIndex,endIndex of arg2 in the sentence.
        """
        fields = line.split("\t")

        if len(fields) == 6:
            sentence_string, verb_string, entity_string, state_change_label, arg1_string, arg2_string = fields
            sentence = sentence_string.split("####")
            verb_parts = verb_string.split(",")
            verb = (int(verb_parts[0]), int(verb_parts[1]))
            entity_parts = entity_string.split(",")
            entity = (int(entity_parts[0]), int(entity_parts[1]))
            arg1_parts = arg1_string.split(",")
            arg1 = (int(arg1_parts[0]), int(arg1_parts[1]))
            arg2_parts = arg2_string.split(",")
            arg2 = (int(arg2_parts[0]), int(arg2_parts[1]))
            return cls(sentence, verb, entity, state_change_label, arg1, arg2)
        else:
            raise RuntimeError("Unrecognized line format: " + line)


class IndexedVerbSemanticsInstance(IndexedInstance):

    def __init__(self,
                 sentence: List[int],
                 verb: List[int],
                 entity: List[int],
                 label: Tuple[List[int], List[List[int]]],
                 index: int=None):

        super(IndexedVerbSemanticsInstance, self).__init__(label, index)
        self.sentence = sentence
        self.verb = verb
        self.entity = entity
        self.label = label

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedVerbSemanticsInstance([], [], [], label=([], [[]]), index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        """
        This simple IndexedVerbSemanticsInstance only has one padding dimension: word_indices.
        """
        return self._get_word_sequence_lengths(self.sentence)

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        self.sentence = self.pad_word_sequence(self.sentence, padding_lengths,
                                               truncate_from_right=False)
        sentence_length = padding_lengths['num_sentence_words']
        self.verb = self.pad_sequence_to_length(self.verb, sentence_length,
                                                truncate_from_right=False)
        self.entity = self.pad_sequence_to_length(self.entity, sentence_length,
                                                  truncate_from_right=False)
        padded_tags = self.pad_sequence_to_length(self.label[1], sentence_length,
                                                  default_value=lambda: self.label[1][0],
                                                  truncate_from_right=False)
        self.label = self.label[0], padded_tags

    @overrides
    def as_training_data(self):
        word_array = numpy.asarray(self.sentence, dtype='int32')
        verb_array = numpy.asarray(self.verb, dtype='int32')
        entity_array = numpy.asarray(self.entity, dtype='int32')

        state_array = numpy.asarray(self.label[0], dtype='int32')
        tag_array = numpy.asarray(self.label[1], dtype='int32')

        return (word_array, verb_array, entity_array), (state_array, tag_array)
