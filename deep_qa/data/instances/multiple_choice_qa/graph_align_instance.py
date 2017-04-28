from typing import Dict, List

import numpy
from overrides import overrides

from deep_qa.data.instances.instance import TextInstance, IndexedInstance
from deep_qa.data.data_indexer import DataIndexer
from deep_qa.common.checks import ConfigurationError

class GraphAlignInstance(TextInstance):
    """
        A ``GraphAlignInstance`` is a kind of ``TextInstance`` (for lack of knowing where else to put it)
        that has a list of graphlets for each answer option, each of which contains one or more alignments.
        Each alignment is a set of features (as floats).
        These graphlets are designed to be used rapidly in pre-built TupleMatch systems, which can hopefully
        learn which feature patterns are indicative of a correct answer (i.e., a good alignment to a knowledge
        base sentence).

        Parameters
        ----------
        answer_graphlets: List[List[List[List[float]]]]
            This is a list whose length is equal to the num answer options, and each list element corresponds to an
            answer option.  Each answer has a list of graphlets which come from features that model the alignment
            between that question + answer option and a knowledge base sentence.

        question_text: str, default=None
            The original text of the question, if available.

        label: int, default=None
            The class label (i.e. the index of the correct multiple choice answer) -- corresponds to the
            indices in self.answer_tuples.

        index: int, default=None
            The index of the question.
        """
    def __init__(self, answer_graphlets: List[List[List[List[float]]]], question_text: str=None,
                 label: int=None, index: int=None):
        super(GraphAlignInstance, self).__init__(label, index)
        self.answer_graphlets = answer_graphlets
        self.question_text = question_text

    @overrides
    def words(self) -> Dict[str, List[str]]:
        # Unused
        return {}

    @overrides
    def to_indexed_instance(self, data_indexer: DataIndexer):
        return IndexedGraphAlignInstance(self.answer_graphlets, self.label, self.index)

    @classmethod
    def read_from_line(cls, line: str):
        """
        Reads a TupleInstances from a line.  The format has two options:

        (1) [question index][tab][all answer graphlets][tab][background tuples][tab][label]
        (2) same as option 1, but [question text][tab] comes immediately after the question index,
            following the tab.

        The answer graphlets are formatted as:

        - [a_1-graphlet_1]$$$...$$$[a_1-graphlet_n]###...###[a+_m-tuple_1]$$$...$$$[a_m-tuple_p]

        such that ``$$$`` serves as the graphlet delimiter and ``###`` serves as the answer candidate
        delimiter.
        Each of the graphlets is formatted as:

        - [alignment1]<>...<>[alignment_m]

        such that ``<>`` serves as the alignment delimiter.  Each alignment contains a series of float
        features, comma separated (e.g., '0.33,0.78,...,0.42')
        """
        fields = line.split("\t")
        if len(fields) == 5:
            index, question_text, answers_string, _, label = fields
            index = int(index)
        elif len(fields) == 4:
            index, answers_string, _, label = fields
            question_text = None
            index = int(index)
        elif len(fields) == 3:
            answers_string, _, label = fields
            question_text = None
            index = None
        else:
            raise RuntimeError("Unrecognized line format (" + str(len(fields)) + " columns): " + line)

        answers = answers_string.split("###")
        answer_graphlets = []
        for answer in answers:
            # Split out the graphlets for the answer
            graphlet_strings = answer.split("$$$")
            graphlets_for_answer = []
            # Split the alignments
            for graphlet_string in graphlet_strings:
                if graphlet_string:
                    alignments = []
                    alignments_string = graphlet_string.split("<>")
                    for alignment_string in alignments_string:
                        alignment_string = alignment_string.rstrip()
                        if alignment_string:
                            alignment = [float(feature) for feature in alignment_string.rstrip().split(",")]
                            alignments.append(alignment)
                    graphlets_for_answer.append(alignments)
            answer_graphlets.append(graphlets_for_answer)

        label = int(label)
        if label >= len(answer_graphlets):
            raise ConfigurationError("Invalid label, label {0} is >= the number "
                                     "of answers ({1}).".format(label, len(answer_graphlets)))

        return cls(answer_graphlets, question_text, label=label, index=index)

class IndexedGraphAlignInstance(IndexedInstance):
    def __init__(self,
                 # (num_answer_options, num_answer_graphlets, num_graphlet_alignments, num_features)
                 answers_indexed: List[List[List[List[float]]]],
                 label,
                 index: int=None):
        super(IndexedGraphAlignInstance, self).__init__(label, index)
        self.answers_indexed = answers_indexed

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedGraphAlignInstance([], label=None, index=None)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        # We care only about the longest slot here because all slots will be padded to the same length.
        max_num_graphlets = 0
        max_num_alignments = 0
        max_num_features = 0
        for answer in self.answers_indexed:
            max_num_graphlets = max(max_num_graphlets, len(answer))
            answer_alignments, answer_features = self.get_lengths_from_indexed_graphlets(answer)
            max_num_alignments = max(max_num_alignments, answer_alignments)
            max_num_features = max(max_num_features, answer_features)

        lengths = {'num_options': len(self.answers_indexed),
                   'num_graphlets': max_num_graphlets,
                   'num_alignments': max_num_alignments,
                   'num_features': max_num_features}
        return lengths

    @staticmethod
    def get_lengths_from_indexed_graphlets(indexed_graphlets: List[List[List[int]]]):
        '''
        Helper method for ``self.get_lengths()``. This gets the max lengths (max
        number alignments, and features per alignment for a list of indexed
        graphlets.

        Parameters
        ----------
        indexed_graphlets: List[List[List[float]]]
            The indexed graphlets that we want to get max lengths from.

        Returns
        -------
        max_num_alignments: int
            The max number of slots found.

        max_num_features: int
            The max number of words found in any of the slots.
        '''
        max_num_alignments = 0
        max_num_features = 0
        for indexed_graphlet in indexed_graphlets:
            num_alignemtns = len(indexed_graphlet)
            max_num_alignments = max(max_num_alignments, num_alignemtns)
            for alignment in indexed_graphlet:
                num_features = len(alignment)
                max_num_features = max(max_num_features, num_features)

        return max_num_alignments, max_num_features

    @overrides
    def pad(self, padding_lengths: Dict[str, int]):
        """
        Pads (or truncates) each indexed graphlet in answers_indexed to specified
        lengths, using the superclass methods ``pad_sequence_to_length`` (for padding
        number of options, number of graphlets, number of alignments per graphlet, and
        number of features per alignment.

        When padding the number of answer options, if there are more answers
        provided, we don't remove options and we raise an exception because
        we don't want to risk removing the correct answer in a multiple choice
        setting.
        """
        desired_num_options = padding_lengths['num_options']
        desired_num_graphlets = padding_lengths['num_graphlets']

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
        # Pad the number of graphlets per option.
        for answer_index in range(len(self.answers_indexed)):
            self.answers_indexed[answer_index] = self.pad_sequence_to_length(self.answers_indexed[answer_index],
                                                                             desired_num_graphlets,
                                                                             lambda: [],
                                                                             truncate_from_right=False)
        # Pad each of the graphlets for num alignments and features per alignment
        for answer_index in range(len(self.answers_indexed)):
            self.answers_indexed[answer_index] = [self.pad_graphlet(answer_graphlet, padding_lengths)
                                                  for answer_graphlet in self.answers_indexed[answer_index]]

    def pad_graphlet(self, graphlet: List[List[int]], padding_lengths: Dict):
        """
        Helper method to pad an individual graphlet both to the desired number of alignments
        as well as the desired features per alignment, both of which are done through the
        superclass method ``pad_sequence_to_length``.

        Parameters
        ----------
        graphlet: List[List[float]]
            The indexed tuple to be padded. This has the shape
            ``(num_alignments, num_features)``.

        padding_lengths: Dict of {str: int}
            The lengths to pad to.  Must include the keys:

            - 'num_alignments': the number of slots desired
            - 'num_features': the number of words in a given slot

        Returns
        -------
        graphlet: List[List[float]]
            In the returned (modified) list, the length matches the desired_num_alignments and each of
            the alignments has a number of features equal to the value set by ``num_features`` in padding_lengths.
        """
        desired_num_alignments = padding_lengths['num_alignments']
        graphlet = self.pad_sequence_to_length(graphlet, desired_num_alignments,
                                               default_value=lambda: [], truncate_from_right=False)
        # Pad the alignment to the desired number of features.
        desired_num_features = padding_lengths['num_features']
        graphlet = [self.pad_sequence_to_length(alignment, desired_num_features, default_value=lambda: 0.0,
                                                truncate_from_right=False) for alignment in graphlet]
        return graphlet

    @overrides
    def as_training_data(self):
        # Question Input:
        # (num_options, num_option_graphlets, num_alignments, num_features)
        question_options_matrix = numpy.asarray(self.answers_indexed, dtype='float32')

        # Represent the label as a class label.
        if self.label is None:
            label = None
        else:
            label = numpy.zeros((len(self.answers_indexed)))
            label[self.label] = 1
        return question_options_matrix, label
