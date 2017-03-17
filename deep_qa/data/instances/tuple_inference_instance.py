from collections import defaultdict
from typing import Dict, List

import numpy as np
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from ..data_indexer import DataIndexer
from ...common.checks import ConfigurationError


class TextTuple:
    def __init__(self, subject: str, verb: str, objects: List[str],
                 context: str=None, time: str=None, location: str=None, source: str=None):
        self.subject = subject
        self.verb = verb
        self.objects = objects
        self.context = context
        self.time = time
        self.location = location
        self.source = source

    def display_string(self, context_limit=None):
        # NOTE: not currently displaying location, time, and source info.
        if context_limit is None:
            context_limit = len(self.context)
        return ("(S: {0}, V: {1}, O: {2}, C: {3})").format(self.subject,
                                                           self.verb,
                                                           ", ".join(self.objects),
                                                           self.context[:context_limit])

    def to_text_list(self, object_handling: str="collapse", include_context: bool=True):
        '''
        This method converts a TextTuple into a list of strings, where each string is a tuple slot or phrase.

        Parameters
        ----------
        object_handling: str, default='collapse'
            Since we can have a variable number of objects, there are several ways to handle them.
            Currently, two options are inplemented:
                - "collapse": collapse all the objects strings into one long string.
                - "first": ignore all but the first object.

        include_context: bool, default=True
            Determines whether or not to include a tuple slot for context information.

        Returns
        -------
        text_list: List[str]
            The list of strings representing the slots in the TextTuple.
        '''
        # Subject and verb
        text_list = [self.subject, self.verb]
        # Object(s):
        if object_handling == "collapse":
            text_list.append(" ".join(self.objects))
        elif object_handling == "first":
            if len(self.objects) > 0:
                text_list.append(self.objects[0])
        else:
            raise ConfigurationError(("Requested object_handling ({0}) not currently " +
                                      "supported.").format(object_handling))
        # Context
        if include_context and self.context:
            text_list.append(self.context)
        # Currently, source, time and location aren't being used, but this can be changed in the future.

        return text_list


    @classmethod
    def from_string(cls, input_string: str):
        """
        Converts a string representation of the tuple into a TextTuple.
        Parameters
        ----------
        input_string: str

        Returns
        -------
        TextTuple
        """
        tuple_fields = input_string.lower().split("<>")
        num_fields = len(tuple_fields)
        if num_fields < 2:
            raise ConfigurationError("num_fields (i.e. num tuple fields) not >= 2.  input_string: " + input_string)
        subject = tuple_fields[0]
        verb = tuple_fields[1]
        objects = []
        context = time = location = source = None
        for element in tuple_fields[2:]:
            if element.startswith("context:"):
                context = element[8:]
            elif element.startswith("l:"):
                location = element[2:]
            elif element.startswith("t:"):
                time = element[2:]
            elif element.startswith("source:"):
                source = element[7:]
            else:
                objects.append(element)
        return TextTuple(subject, verb, objects, context, time, location, source)


class TupleInferenceInstance(TextInstance):
    """
    A ``TupleInferenceInstance`` is a kind of ``TextInstance`` that has text in multiple slots or phrases
    (stored as a ``TextTuple``), for both the answer choices and background information.
    This can be used to store SVO triples.  These tuples are designed to be used in systems which
    compare question-answer tuples to background knowledge tuples, to preform a sort of "inference" about
    whether or not the question is answered correctly.

    Parameters
    ----------
    answer_tuples: List[List[TextTuple]]
        This is a list whose length is equal to the num answer options, and each list element corresponds to an
        answer option.  Each answer has a list of ``TextTuples`` which come from combining that answer option
        with the question.

    background_tuples: List[TextTuple]
        This is a list of background ``TextTuples`` (currently used for all answer candidates).

    question_text: str, default=None
        The original text of the question, if available.

    label: int, default=None
        The class label (i.e. the index of the correct multiple choice answer) -- corresponds to the
        indices in self.answer_tuples.

    index: int, default=None
        The index of the question.
    """
    object_handling = 'collapse'
    use_context = True

    def __init__(self, answer_tuples: List[List[TextTuple]], background_tuples: List[TextTuple],
                 question_text: str=None, label: int=None, index: int=None):
        super(TupleInferenceInstance, self).__init__(label, index)
        self.answer_tuples = answer_tuples
        self.background_tuples = background_tuples
        self.question_text = question_text

    def display_string(self):
        to_return = 'TupleInferenceInstance: \n'
        to_return += "Answer Candidates: \n"
        for answer_index in range(len(self.answer_tuples)):
            string_answer_tuples = [a_tuple.display_string() for a_tuple in self.answer_tuples[answer_index]]
            if answer_index == self.label:
                to_return += "**"   # To inidicate correctness in a visually easy to parse way
            else:
                to_return += "  "
                to_return += "Answer Option [{0}]:\n{1}\n".format(answer_index, ",\n".join(string_answer_tuples))
            to_return += "\nBackground Tuples:\n"
            for background_tuple in self.background_tuples:
                to_return += background_tuple.display_string() + ",\n"
        return to_return

    @overrides
    def words(self) -> Dict[str, List[str]]:
        words = defaultdict(list)
        all_tuples = self.background_tuples + [curr_tuple for answer in self.answer_tuples
                                               for curr_tuple in answer]
        for text_tuple in all_tuples:
            for phrase in text_tuple.to_text_list(self.object_handling,
                                                  self.use_context):
                phrase_words = self._words_from_text(phrase)
                for namespace in phrase_words:
                    words[namespace].extend(phrase_words[namespace])
        return words

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        answers_indexed = [self.index_tuples(curr_answer_tuples, data_indexer) for
                           curr_answer_tuples in self.answer_tuples]
        background_indexed = self.index_tuples(self.background_tuples, data_indexer)
        return IndexedTupleInferenceInstance(answers_indexed,
                                             background_indexed,
                                             self.label,
                                             self.index)

    def index_tuples(self, tuples_in: List[TextTuple], data_indexer: DataIndexer):
        """
        This is a helper method for to_indexed_instance which indexes a list of ``TextTuple``.
        Parameters
        ----------
        tuples_in: List[TextTuple]
            A list of ``TextTuple`` to be indexed.

        data_indexer: DataIndexer

        Returns
        -------
        indexed: List[List[List[int]]]
            The indexed tuple.
        """
        indexed = []
        for text_tuple in tuples_in:
            text_list = text_tuple.to_text_list(self.object_handling, self.use_context)
            # indices: List[List[int]]
            indices = [self._index_text(phrase, data_indexer) for phrase in text_list]
            indexed.append(indices)
        return indexed

    @classmethod
    def read_from_line(cls, line: str, default_label: int=None):
        """
        Reads a TupleInferenceInstance from a line.  The format has two options (currently):
            Option 1:
                [question index][tab][all question+answer tuples][tab][background tuples][tab][label]
            Option 2:
                same as option 1, but [question text][tab] comes immediately after the question index,
                following the tab.
        The question+answer tuples are formatted as:
            [a_1-tuple_1]$$$...$$$[a_1-tuple_n]###...###[a+_m-tuple_1]$$$...$$$[a_m-tuple_p]
        such that ``$$$`` serves as the tuple delimiter and ``###`` serves as the answer candidate
        delimiter.
        Each of the tuples is formatted as:
            [s words]<>[v words]<>[o_1 words]<>...<>[o_j words]<>[context: context words]
        such that ``<>`` serves as the tuple-slot delimiter.  Note that some slots may contain prefixes
        indicating the type of information they contain (e.g., "context:", see below for more).

        Objects are optional, and can vary in number. This makes the acceptable number of slots per
        tuple 2 or more.

        If a tuple slot is reserved for context, it contains "context:" at the start.

        Some tuples contain coreference resolution info (from one of the table sources).  We do not
        currently handle this, but plan to eventually replace the referring expression with its resolved
        value.
            Example: it[sky]<>is<>blue --> would become --> sky<>is<>blue

        Some tuples additionally contain prefixed information about some of the "objects" such as "L:" for
        locations and T: for time.  At present this information is simply filtered out, but can be used
        later.
        """
        fields = line.split("\t")
        if len(fields) == 5:
            index, question_text, answers_string, background_string, label = fields
            index = int(index)
        elif len(fields) == 4:
            index, answers_string, background_string, label = fields
            question_text = None
            index = int(index)
        elif len(fields) == 3:
            answers_string, background_string, label = fields
            question_text = None
            index = None
        else:
            raise RuntimeError("Unrecognized line format (" + str(len(fields)) + " columns): " + line)

        answers = answers_string.split("###")
        answer_tuples = []
        for answer in answers:
            tuple_strings = answer.split("$$$")
            tuples = [TextTuple.from_string(tuple_string) for tuple_string in tuple_strings if tuple_string]
            answer_tuples.append(tuples)

        background_tuple_strings = background_string.split("$$$")
        background_tuples = [TextTuple.from_string(background_string)
                             for background_string in background_tuple_strings if background_string]
        label = int(label)
        if label >= len(answer_tuples):
            raise ConfigurationError("Invalid label, label is >= the number of answers.")

        return cls(answer_tuples, background_tuples, question_text, label=label, index=index)


class IndexedTupleInferenceInstance(IndexedInstance):
    def __init__(self,
                 # (num_answer_options, num_answer_tuples, num_tuple_slots, num_slot_words)
                 answers_indexed: List[List[List[List[int]]]],
                 # (num_background_tuples, num_tuple_slots, num_slot_words)
                 background_indexed: List[List[List[int]]],
                 label,
                 index: int=None):
        super(IndexedTupleInferenceInstance, self).__init__(label, index)
        self.answers_indexed = answers_indexed
        self.background_indexed = background_indexed

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedTupleInferenceInstance([], [], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        # We care only about the longest slot here because all slots will be padded to the same length.
        max_slot_length = 0
        num_word_characters = 0
        max_num_slots = 0
        max_question_tuples = 0
        for answer in self.answers_indexed:
            answer_slots, answer_length, answer_word_length = self.get_lengths_from_indexed_tuples(answer)
            max_num_slots = max(max_num_slots, answer_slots)
            max_slot_length = max(max_slot_length, answer_length)
            num_word_characters = max(num_word_characters, answer_word_length)
            max_question_tuples = max(max_question_tuples, len(answer))
        background_slots, background_length, background_word_length = \
            self.get_lengths_from_indexed_tuples(self.background_indexed)
        max_num_slots = max(max_num_slots, background_slots)
        max_slot_length = max(max_slot_length, background_length)
        num_word_characters = max(num_word_characters, background_word_length)

        lengths = {'num_options': len(self.answers_indexed),
                   'num_question_tuples': max_question_tuples,
                   'num_background_tuples': len(self.background_indexed),
                   'num_sentence_words': max_slot_length,
                   'num_slots': max_num_slots}
        if num_word_characters > 0:
            lengths['num_word_characters'] = num_word_characters
        return lengths

    @staticmethod
    def get_lengths_from_indexed_tuples(indexed_tuples: List[List[List[int]]]):
        '''
        Helper method for self.get_lengths().  This gets the max lengths (max number of slots and max
        number of words per slot) for a list of indexed tuples.

        Parameters
        ----------
        indexed_tuples: List[List[List[int]]]
            The indexed tuples that we want to get max lengths from.

        Returns
        -------
        max_num_slots: int
            The max number of slots found.

        max_slot_words: int
            The max number of words found in any of the slots.

        num_word_characters: int
            The length of the longest word, with a return value of 0 if the model isn't using character padding.
        '''
        max_num_slots = 0
        max_slot_words = 0
        num_word_characters = 0
        for indexed_tuple in indexed_tuples:
            num_slots = len(indexed_tuple)
            max_num_slots = max(max_num_slots, num_slots)
            for slot in indexed_tuple:
                num_sentence_wordss = IndexedInstance._get_word_sequence_lengths(slot)
                max_slot_words = max(max_slot_words, num_sentence_wordss['num_sentence_words'])
                if 'num_word_characters' in num_sentence_wordss:
                    num_word_characters = max(num_word_characters, num_sentence_wordss['num_word_characters'])

        return max_num_slots, max_slot_words, num_word_characters

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) each indexed tuple in answers_indexed and background_indexed to specified lengths,
        using the superclass methods ``pad_sequence_to_length`` (for padding number of options, number of tuples,
        and number of slots per tuple) and ''pad_word_sequence`` (for padding the words in each slot), with the
        following exception:
            When padding the number of answer options, if there are more answers provided, here we don't
            remove options, we raise an exception because we don't want to risk removing the correct answer in
            a multiple choice setting.
        """
        desired_num_options = max_lengths['num_options']
        desired_num_question_tuples = max_lengths['num_question_tuples']
        desired_num_background_tuples = max_lengths['num_background_tuples']

        # Pad the number of answers. Note that while we use the superclass method when we need to add empty
        # answer_options, we don't want to remove answer options.
        num_options = len(self.answers_indexed)
        if num_options > desired_num_options:
            raise Exception("Case of too many answer options ({0} provided) isn't currently "
                            "handled.".format(num_options))
        elif num_options < desired_num_options:
            # Add empty answer option(s).
            self.answers_indexed = self.pad_sequence_to_length(self.answers_indexed,
                                                               desired_num_options,
                                                               lambda: [],
                                                               truncate_from_right=False)
        # Pad the number of tuples per option.
        for answer_index in range(len(self.answers_indexed)):
            self.answers_indexed[answer_index] = self.pad_sequence_to_length(self.answers_indexed[answer_index],
                                                                             desired_num_question_tuples,
                                                                             lambda: [],
                                                                             truncate_from_right=False)
        # Pad each of the tuples for slot length and words per slot
        for answer_index in range(len(self.answers_indexed)):
            self.answers_indexed[answer_index] = [self.pad_tuple(answer_tuple, max_lengths)
                                                  for answer_tuple in self.answers_indexed[answer_index]]
        # Pad the number of background tuples
        self.background_indexed = self.pad_sequence_to_length(self.background_indexed,
                                                              desired_num_background_tuples,
                                                              lambda: [],
                                                              truncate_from_right=False)
        # Pad each of the background tuples for number of slots and slot length
        self.background_indexed = [self.pad_tuple(background_tuple, max_lengths)
                                   for background_tuple in self.background_indexed]

    def pad_tuple(self, tuple_in: List[List[int]], max_lengths: Dict):
        '''
        Helper method to pad an individual tuple both to the desired number of slots as well as the
        desired slot length, both of which are done through the superclass method ``pad_word_sequence``.

        Parameters
        ----------
        tuple_in: List[List[int]]
            (num_tuple_slots, num_slot_words)
            The indexed tuple to be padded.

        max_lengths: Dict of {str: int}
            The lengths to pad to.  Must include the keys:
                - 'num_slots': the number of slots desired
                - 'num_sentence_words': the number of words in a given slot
            May also include:
                - 'num_word_characters': the length of each word,
                relevant when using a ``WordAndCharacterTokenizer``.

        Returns
        -------
        tuple_in: List[List[int]]
            In the returned (modified) list, the length matches the desired_num_slots and each of the slots
            has a length equal to the value set by 'num_sentence_words' in max_lengths.
        '''
        desired_num_slots = max_lengths['num_slots']
        tuple_in = self.pad_sequence_to_length(tuple_in, desired_num_slots,
                                               default_value=lambda: [], truncate_from_right=False)
        # Pad the slots to the desired length.
        tuple_in = [self.pad_word_sequence(indices, max_lengths) for indices in tuple_in]
        return tuple_in

    @overrides
    def as_training_data(self):
        # Question Input:
        # (num_options, num_option_tuples, num_slots, max_question_slot_length)
        question_options_matrix = np.asarray(self.answers_indexed, dtype='int32')

        # Background Input:
        # (num_background_tuples, num_slots, max_background_slot_length)
        background_matrix = np.asarray(self.background_indexed, dtype='int32')

        # Represent the label as a class label.
        if self.label is None:
            label = None
        else:
            label = np.zeros((len(self.answers_indexed)))
            label[self.label] = 1

        return (question_options_matrix, background_matrix), label
