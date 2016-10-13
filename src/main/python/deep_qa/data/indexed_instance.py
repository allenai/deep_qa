from typing import Dict, List
from overrides import overrides

import numpy

from .instance import Instance

class IndexedInstance(Instance):
    """
    An indexed data instance has all word tokens replaced with word indices, along with some kind
    of label, suitable for input to a Keras model.  An IndexedInstance is created from an Instance
    using a DataIndexer, and the indices here have no recoverable meaning without the DataIndexer.

    For example, we might have the following instance:
        TrueFalseInstance('Jamie is nice, Holly is mean', True, 25).
    After being converted into an IndexedInstance, we might have the following:
        IndexedTrueFalseInstance([1, 6, 7, 1, 6, 8], True, 25).
    This would mean that "Jamie" and "Holly" were OOV to the DataIndexer, and the other words were
    given indices.
    """
    @classmethod
    def empty_instance(cls):
        """
        Returns an empty, unpadded instance of this class.  Necessary for option padding in
        multiple choice instances.
        """
        raise NotImplementedError

    def get_lengths(self) -> List[int]:
        """
        Used for padding.  Different kinds of instances have different fields that are padded, such
        as sentence length, number of background sentences, number of options, etc.  The length of
        this instance in all dimensions that require padding are returned here.
        """
        raise NotImplementedError

    def pad(self, max_lengths: List[int]):
        """
        The max_lengths argument passed here must have the same dimension as was returned by
        get_lengths().  We will use these lengths to pad the instance in all of the necessary
        dimensions to the given lengths.

        This modifies the current object.
        """
        raise NotImplementedError

    def as_training_data(self):
        """
        Returns a tuple of (inputs, label).  `inputs` might itself be a complex tuple, depending on
        the Instance type.
        """
        raise NotImplementedError

    @staticmethod
    def pad_word_sequence_to_length(indices: List[int], desired_length: int) -> List[int]:
        """
        Take a list of word indices and pads them to the desired length.

        If we need to truncate the word indices, we do it from the _right_, not the left.  This is
        important for cases that are questions, with long set ups.  We at least want to get the
        question encoded, which is always at the end, even if we've lost much of the question set
        up.

        Though, this might be backwards for encoding background information...
        """
        padded_indices = [0] * desired_length
        indices_length = min(len(indices), desired_length)
        if indices_length != 0:
            padded_indices[-indices_length:] = indices[-indices_length:]
        return padded_indices


class IndexedTrueFalseInstance(IndexedInstance):
    def __init__(self, word_indices: List[int], label, index: int=None):
        super(IndexedTrueFalseInstance, self).__init__(label, index)
        self.word_indices = word_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedTrueFalseInstance([], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        This simple IndexedInstance only has one padding dimension: word_indices.
        """
        return {'word_sequence_length': len(self.word_indices)}

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) self.word_indices to be of length max_lengths[0].  See comment on
        self.get_lengths() for why max_lengths is a list instead of an int.
        """
        desired_length = max_lengths['word_sequence_length']
        self.word_indices = self.pad_word_sequence_to_length(self.word_indices, desired_length)

    @overrides
    def as_training_data(self):
        word_array = numpy.asarray(self.word_indices, dtype='int32')
        if self.label is True:
            label = numpy.zeros((2))
            label[1] = 1
        elif self.label is False:
            label = numpy.zeros((2))
            label[0] = 1
        else:
            label = None
        return word_array, label


class IndexedLogicalFormInstance(IndexedTrueFalseInstance):
    """
    An IndexedLogicalFormInstance is a tree-structured instance, which represents a logical form
    like "for(depend_on(human, plant), oxygen)" as a pair of: (1) a (sequential) list of predicates
    and arguments, and (2) a list of shift/reduce operations, which allows recovery of the original
    tree structure from the sequential list of predicates and arguments.  This allows us to do tree
    composition in a compiled neural network - we just have to pad to the maximum transition
    length, and we can represent arbitrarily shaped trees.

    Idea taken from the SPINN paper by Sam Bowman and others (http://arxiv.org/pdf/1603.06021.pdf).
    """
    def __init__(self, word_indices: List[int], transitions: List[int], label: bool, index: int=None):
        super(IndexedLogicalFormInstance, self).__init__(word_indices, label, index)
        self.transitions = transitions

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedLogicalFormInstance([], [], None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for `transitions`.
        """
        lengths = super(IndexedLogicalFormInstance, self).get_lengths()
        lengths['transition_length'] = len(self.transitions)
        return lengths

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        We let the super class deal with padding word_indices; we'll worry about padding
        transitions.
        """
        super(IndexedLogicalFormInstance, self).pad(max_lengths)

        transition_length = max_lengths['transition_length']
        self.transitions = self.pad_word_sequence_to_length(self.transitions, transition_length)

    @overrides
    def as_training_data(self):
        word_array, label = super(IndexedLogicalFormInstance, self).as_training_data()
        transitions = numpy.asarray(self.transitions, dtype='int32')
        return (word_array, transitions), label


class IndexedBackgroundInstance(IndexedInstance):
    """
    An IndexedInstance that has background knowledge associated with it, where the background
    knowledge has also been indexed.
    """
    contained_instance_type = None
    def __init__(self,
                 indexed_instance: IndexedInstance,
                 background_indices: List[List[int]]):
        super(IndexedBackgroundInstance, self).__init__(indexed_instance.label, indexed_instance.index)
        self.indexed_instance = indexed_instance
        self.background_indices = background_indices

        # We need to set this here so that we know what kind of contained instance we should create
        # when we're asked for an empty IndexedBackgroundInstance.  Note that this assumes that
        # you'll only ever have one underlying Instance type, which is a reasonable assumption
        # given our current code.
        IndexedBackgroundInstance.contained_instance_type = indexed_instance.__class__

    @classmethod
    @overrides
    def empty_instance(cls):
        contained_instance = IndexedBackgroundInstance.contained_instance_type.empty_instance()
        return IndexedBackgroundInstance(contained_instance, [])

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Prep for padding; see comment on this method in the super class.  Here we extend the return
        value from our super class with the padding lengths necessary for background_indices.

        Additionally, as we currently use the same encoder for both a sentence and its background
        knowledge, we'll also modify the word_indices length to look at the background sentences
        too.
        """
        lengths = self.indexed_instance.get_lengths()
        lengths['background_sentences'] = len(self.background_indices)
        if self.background_indices:
            max_background_length = max(len(background) for background in self.background_indices)
            lengths['word_sequence_length'] = max(lengths['word_sequence_length'], max_background_length)
        return lengths

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        We let self.indexed_instance pad itself, and in this method we mostly worry about padding
        background_indices.  We need to pad it in two ways: (1) we need len(background_indices) to
        be the same for all instances, and (2) we need len(background_indices[i]) to be the same
        for all i, for all instances.  We'll use the word_indices length from the super class for
        (2).
        """
        self.indexed_instance.pad(max_lengths)
        background_length = max_lengths['background_sentences']
        word_sequence_length = max_lengths['word_sequence_length']

        # Padding (1): making sure we have the right number of background sentences.  We also need
        # to truncate, if necessary.
        if len(self.background_indices) > background_length:
            self.background_indices = self.background_indices[:background_length]
        for _ in range(background_length - len(self.background_indices)):
            self.background_indices.append([0])

        # Padding (2): making sure all background sentences have the right length.
        padded_background = []
        for background in self.background_indices:
            padded_background.append(self.pad_word_sequence_to_length(background, word_sequence_length))
        self.background_indices = padded_background

    @overrides
    def as_training_data(self):
        """
        This returns a complex output.  In the simplest case, the contained instance is just a
        TrueFalseInstance, with a single sentence input.  In this case, we'll return a tuple of
        (sentence_array, background_array) as the inputs (and, as always, the label from the
        contained instance).

        If the contained instance itself has multiple inputs it returns, we need the
        background_array to be second in the list (because that makes the implementation in the
        memory network solver much easier).  That means we need to change the order of things
        around a bit.
        """
        instance_inputs, label = self.indexed_instance.as_training_data()
        background_array = numpy.asarray(self.background_indices, dtype='int32')
        if isinstance(instance_inputs, tuple):
            final_inputs = (instance_inputs[0],) + (background_array,) + instance_inputs[1:]
        else:
            final_inputs = (instance_inputs, background_array)
        return final_inputs, label


class IndexedLabeledBackgroundInstance(IndexedBackgroundInstance):
    """
    This is an IndexedBackgroundInstance that has a different label.  Instead of passing through
    the contained instance's label, we have labeled attention over the background data.  So this
    object behaves identically to IndexedBackgroundInstance in everything except the label.

    See text_instance.LabeledBackgroundInstance for a little more detail.
    """
    def __init__(self,
                 indexed_instance: IndexedInstance,
                 background_indices: List[List[int]],
                 label: List[int]):
        super(IndexedLabeledBackgroundInstance, self).__init__(indexed_instance, background_indices)
        self.label = label

    @overrides
    def as_training_data(self):
        """
        All we do here is overwrite the label from IndexedBackgroundInstance.
        """
        inputs, _ = super(IndexedLabeledBackgroundInstance, self).as_training_data()
        if self.label is None:
            label = None
        else:
            label = numpy.zeros(len(self.background_indices))
            for index in self.label:
                label[index] = 1
        return inputs, label


class IndexedMultipleChoiceInstance(IndexedInstance):
    """
    A MultipleChoiceInstance that has been indexed.  MultipleChoiceInstance has a better
    description of what this represents.
    """
    def __init__(self, options: List[IndexedInstance], label):
        super(IndexedMultipleChoiceInstance, self).__init__(label=label, index=None)
        self.options = options

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedMultipleChoiceInstance([], None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Here we return the max of get_lengths on all of the Instances in self.options.
        """
        max_lengths = {}
        max_lengths['num_options'] = len(self.options)
        lengths = [instance.get_lengths() for instance in self.options]
        if not lengths:
            return max_lengths
        for key in lengths[0]:
            max_lengths[key] = max(x[key] for x in lengths)
        return max_lengths

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        This method pads all of the underlying Instances in self.options.
        """
        num_options = max_lengths['num_options']

        # First we pad the number of options.
        while len(self.options) < num_options:
            self.options.append(self.options[0].empty_instance())
        self.options = self.options[:num_options]

        # Then we pad each option.
        for instance in self.options:  # type: IndexedInstance
            instance.pad(max_lengths)

    @overrides
    def as_training_data(self):
        inputs = []
        unzip_inputs = False
        for option in self.options:
            option_input, _ = option.as_training_data()
            if isinstance(option_input, tuple):
                unzip_inputs = True
            inputs.append(option_input)
        if unzip_inputs:
            inputs = tuple(zip(*inputs))  # pylint: disable=redefined-variable-type
            inputs = tuple([numpy.asarray(x) for x in inputs])
        else:
            inputs = numpy.asarray(inputs)
        if self.label is None:
            label = None
        else:
            label = numpy.zeros(len(self.options))
            label[self.label] = 1
        return inputs, label


class IndexedQuestionAnswerInstance(IndexedInstance):
    def __init__(self,
                 question_indices: List[int],
                 option_indices: List[List[int]],
                 label: int,
                 index: int=None):
        super(IndexedQuestionAnswerInstance, self).__init__(label, index)
        self.question_indices = question_indices
        self.option_indices = option_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedQuestionAnswerInstance([], [], 0)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        """
        Three things to pad here: the question length, the answer option length, and the number of
        answer options.
        """
        question_length = len(self.question_indices)
        max_answer_length = max([len(indices) for indices in self.option_indices])
        num_options = len(self.option_indices)
        return {
                'word_sequence_length': question_length,
                'answer_length': max_answer_length,
                'num_options': num_options,
                }

    @overrides
    def pad(self, max_lengths: List[int]):
        """
        Three things to pad here: the question length, the answer option length, and the number of
        answer options.
        """
        question_length = max_lengths['word_sequence_length']
        answer_length = max_lengths['answer_length']
        num_options = max_lengths['num_options']
        self.question_indices = self.pad_word_sequence_to_length(self.question_indices, question_length)

        while len(self.option_indices) < num_options:
            self.option_indices.append([])
        self.option_indices = self.option_indices[:num_options]

        padded_options = []
        for indices in self.option_indices:
            padded_options.append(self.pad_word_sequence_to_length(indices, answer_length))
        self.option_indices = padded_options

    @overrides
    def as_training_data(self):
        question_array = numpy.asarray(self.question_indices, dtype='int32')
        option_array = numpy.asarray(self.option_indices, dtype='int32')
        if self.label is None:
            label = None
        else:
            label = numpy.zeros((len(self.option_indices)))
            label[self.label] = 1
        return (question_array, option_array), label


class IndexedSentencePairInstance(IndexedInstance):
    """
    This is an indexed instance that is commonly used for labeled sentence pairs. Examples of this are
    SnliInstances where we have a labeled pair of text and hypothesis, and a sentence2vec instance where the
    objective is to train an encoder to predict whether the sentences are in context or not.
    """
    def __init__(self, first_sentence_indices: List[int], second_sentence_indices: List[int], label: List[int],
                 index: int=None):
        super(IndexedSentencePairInstance, self).__init__(label, index)
        self.first_sentence_indices = first_sentence_indices
        self.second_sentence_indices = second_sentence_indices

    @classmethod
    @overrides
    def empty_instance(cls):
        return IndexedSentencePairInstance([], [], label=None, index=None)

    @overrides
    def get_lengths(self) -> Dict[str, int]:
        max_length = max(len(self.first_sentence_indices), len(self.second_sentence_indices))
        return {'word_sequence_length': max_length}

    @overrides
    def pad(self, max_lengths: Dict[str, int]):
        """
        Pads (or truncates) self.word_indices to be of length max_lengths[0].  See comment on
        self.get_lengths() for why max_lengths is a list instead of an int.
        """
        sentence_length = max_lengths['word_sequence_length']
        self.first_sentence_indices = self.pad_word_sequence_to_length(self.first_sentence_indices,
                                                                       sentence_length)
        self.second_sentence_indices = self.pad_word_sequence_to_length(self.second_sentence_indices,
                                                                        sentence_length)

    @overrides
    def as_training_data(self):
        first_sentence_array = numpy.asarray(self.first_sentence_indices, dtype='int32')
        second_sentence_array = numpy.asarray(self.second_sentence_indices, dtype='int32')
        return (first_sentence_array, second_sentence_array), numpy.asarray(self.label)
