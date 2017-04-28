# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.instances.multiple_choice_qa.graph_align_instance import GraphAlignInstance
from deep_qa.data.instances.multiple_choice_qa.graph_align_instance import IndexedGraphAlignInstance
from tests.common.test_case import DeepQaTestCase


class TestGraphAlignInstance(DeepQaTestCase):
    def setUp(self):
        super(TestGraphAlignInstance, self).setUp()
        q_idx = 0
        # ans delim: ###
        # graphlet delim: $$$
        # alignment delim: <>
        # Here we have 2 ans, ans 1 has 2 graphlets each with one alignment, ans 2 has 1 graphlet with 2 alignments
        answer_options = ("0.1,0.2$$$0.3,0.4###0.11,0.22<>0.111,0.222")
        label = "0"
        self.line = "\t".join(str(x) for x in [q_idx, answer_options, "NONE", label])
        self.instance = GraphAlignInstance.read_from_line(self.line)

        # No tuples for option 0 or 2 (of 4 options)
        # Should have 4 answers:
        # ans0 has 0 graphlets
        # ans1 has 2 graphlets, each with one alignment
        # ans2 has 0 graphlets
        # ans3 has two graphlets, each with one alignment
        q_idx2 = 1
        answer_options2 = ("###0.2,0.1$$$0.3,0.4######0.2,0.1$$$0.3,0.4")
        label2 = "1"
        line2 = "\t".join(str(x) for x in [q_idx2, answer_options2, "NONE", label2])
        self.instance_2 = GraphAlignInstance.read_from_line(line2)

        # For testing indexed instances
        alignments1 = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
        alignments2 = [[0.3, 0.3, 0.3], [0.4, 0.4, 0.4]]
        answer1 = [alignments1, alignments2]
        answer2 = [alignments1]
        self.indexed_instance = IndexedGraphAlignInstance([answer1, answer2], 2, 0)

    def test_load_from_file_splits_correctly(self):
        # test general case
        # answer_options = ("0.1,0.2$$$0.3,0.4###0.11,0.22<>0.111,0.222")
        # Here we have 2 ans, ans 1 has 2 graphlets each with one alignment, ans 2 has 1 graphlet with 2 alignments
        assert len(self.instance.answer_graphlets) == 2
        assert len(self.instance.answer_graphlets[0]) == 2
        assert len(self.instance.answer_graphlets[1]) == 1
        assert self.instance.label == 0
        assert len(self.instance.answer_graphlets[0][0]) == 1
        assert len(self.instance.answer_graphlets[1][0]) == 2
        # test no tuples for answer option
        assert len(self.instance_2.answer_graphlets) == 4
        assert len(self.instance_2.answer_graphlets[0]) == 0
        assert len(self.instance_2.answer_graphlets[2]) == 0

    def test_indexed_instance_padding(self):
        data_indexer = DataIndexer()
        indexed = self.instance.to_indexed_instance(data_indexer)
        num_graphlets = 1
        num_alignments = 2
        num_features = 5
        num_options = 4
        padding_lengths = {'num_graphlets': num_graphlets,
                           'num_alignments': num_alignments,
                           'num_features': num_features,
                           'num_options': num_options}
        indexed.pad(padding_lengths)
        assert len(indexed.answers_indexed) == num_options
        for answer_option_graphlets in indexed.answers_indexed:
            assert len(answer_option_graphlets) == num_graphlets
            for answer_graphlet in answer_option_graphlets:
                assert len(answer_graphlet) == num_alignments
                for alignment in answer_graphlet:
                    assert len(alignment) == num_features

    def test_as_training_data_produces_correct_numpy_arrays(self):
        padding_lengths = {'num_graphlets': 2,
                           'num_alignments': 2,
                           'num_features': 2,
                           'num_options': 3}
        self.indexed_instance.pad(padding_lengths)

        inputs, label = self.indexed_instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 1]))

        desired_options = numpy.asarray([[[[0.1, 0.1], [0.2, 0.2]],
                                          [[0.3, 0.3], [0.4, 0.4]]],
                                         [[[0.1, 0.1], [0.2, 0.2]],
                                          [[0.0, 0.0], [0.0, 0.0]]],
                                         [[[0.0, 0.0], [0.0, 0.0]],
                                          [[0.0, 0.0], [0.0, 0.0]]]], dtype='float32')

        assert numpy.all([inputs] == desired_options)
