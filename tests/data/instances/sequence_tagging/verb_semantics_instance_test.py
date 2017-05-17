# pylint: disable=no-self-use,invalid-name

from deep_qa.data import DataIndexer
from deep_qa.data.instances.sequence_tagging.verb_semantics_instance import VerbSemanticsInstance

import numpy
from numpy.testing import assert_array_almost_equal
from ....common.test_case import DeepQaTestCase


class TestVerbSemanticsInstance(DeepQaTestCase):
    def setUp(self):
        super(TestVerbSemanticsInstance, self).setUp()
        sentence = ["roots", "absorb", "water"]
        verb = (1, 1)
        entity = (2, 2)
        state_change = "MOVE"
        arg1 = (-1, -1)
        arg2 = (0, 0)
        self.instance = VerbSemanticsInstance(sentence, verb, entity, state_change, arg1, arg2)

        self.data_indexer = DataIndexer()
        self.roots_index = self.data_indexer.add_word_to_index("roots")
        self.absorb_index = self.data_indexer.add_word_to_index("absorb")
        self.water_index = self.data_indexer.add_word_to_index("water")
        self.data_indexer.add_word_to_index("MOVE", namespace="state_changes")
        self.data_indexer.add_word_to_index("O", namespace="tags")
        self.data_indexer.add_word_to_index("B-ARG2", namespace="tags")
        self.expected_sentence_indices = [self.roots_index, self.absorb_index, self.water_index]
        self.b_arg2_tag_one_hot = numpy.asarray([0, 1], dtype='int32')
        self.o_tag_one_hot = numpy.asarray([1, 0], dtype='int32')
        self.expected_tags = [self.b_arg2_tag_one_hot, self.o_tag_one_hot, self.o_tag_one_hot]

        self.indexed_instance = self.instance.to_indexed_instance(self.data_indexer)

    def tearDown(self):
        super(TestVerbSemanticsInstance, self).tearDown()

    def test_read_from_arguments(self):
        assert self.instance.sentence == ["roots", "absorb", "water"]
        assert self.instance.verb == (1, 1)
        assert self.instance.entity == (2, 2)
        assert self.instance.label[0] == "MOVE"
        assert self.instance.label[1] == ["B-ARG2", "O", "O"]

    def test_read_from_line(self):
        line_record = "roots####absorb####water\t1,1\t2,2\tMOVE\t-1,-1\t0,0"
        instance = VerbSemanticsInstance.read_from_line(line_record)

        assert instance.sentence == ["roots", "absorb", "water"]
        assert instance.verb == (1, 1)
        assert instance.entity == (2, 2)
        assert instance.label[0] == "MOVE"
        assert instance.label[1] == ["B-ARG2", "O", "O"]

    def test_convert_instance_to_indexed_instance(self):
        assert_array_almost_equal(self.indexed_instance.sentence, self.expected_sentence_indices)
        assert_array_almost_equal(self.indexed_instance.verb, [0, 1, 0])
        assert_array_almost_equal(self.indexed_instance.entity, [0, 0, 1])

        assert_array_almost_equal(self.indexed_instance.label[0], [1])
        assert_array_almost_equal(self.indexed_instance.label[1], self.expected_tags)

    def test_as_training_data(self):
        train_inputs, train_labels = self.indexed_instance.as_training_data()
        expected_indices = (self.expected_sentence_indices, [0, 1, 0], [0, 0, 1])
        assert_array_almost_equal(train_inputs[0], expected_indices[0])
        assert_array_almost_equal(train_inputs[1], expected_indices[1])
        assert_array_almost_equal(train_inputs[2], expected_indices[2])

        assert_array_almost_equal(train_labels[0], [1])
        assert_array_almost_equal(train_labels[1], self.expected_tags)

    def test_padding(self):
        self.data_indexer.add_word_to_index("from")
        self.data_indexer.add_word_to_index("soil")
        padding_lengths = {'num_sentence_words': 5}
        self.data_indexer.add_word_to_index("CREATE", namespace="state_changes")
        self.data_indexer.add_word_to_index("DESTROY", namespace="state_changes")
        self.data_indexer.add_word_to_index("O", namespace="tags")
        self.data_indexer.add_word_to_index("B-ARG2", namespace="tags")

        self.indexed_instance = self.instance.to_indexed_instance(self.data_indexer)
        self.indexed_instance.pad(padding_lengths)

        expected_sentence_indices = [self.roots_index, self.absorb_index, self.water_index, 0, 0]
        expected_verb_indices = [0, 1, 0, 0, 0]
        expected_entity_indices = [0, 0, 1, 0, 0]
        b_array = numpy.asarray([0, 1], dtype='int32')
        o_array = numpy.asarray([1, 0], dtype='int32')
        expected_label = (numpy.asarray([1, 0, 0], dtype='int32'),
                          numpy.asarray([b_array, o_array, o_array, b_array, b_array], dtype='int32'))
        expected_indices = (expected_sentence_indices, expected_verb_indices, expected_entity_indices)

        assert_array_almost_equal(self.indexed_instance.sentence, expected_indices[0])
        assert_array_almost_equal(self.indexed_instance.verb, expected_indices[1])
        assert_array_almost_equal(self.indexed_instance.entity, expected_indices[2])
        assert_array_almost_equal(self.indexed_instance.label[0], expected_label[0])
        assert_array_almost_equal(self.indexed_instance.label[1], expected_label[1])
