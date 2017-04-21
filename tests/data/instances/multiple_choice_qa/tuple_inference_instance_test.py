# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.data.data_indexer import DataIndexer
from deep_qa.data.dataset import TextDataset
from deep_qa.data.instances.multiple_choice_qa.tuple_inference_instance import TupleInferenceInstance
from deep_qa.data.instances.multiple_choice_qa.tuple_inference_instance import IndexedTupleInferenceInstance
from deep_qa.data.instances.instance import TextInstance
from deep_qa.data.tokenizers import WordAndCharacterTokenizer, tokenizers
from deep_qa.common.params import Params
from ....common.test_case import DeepQaTestCase


class TestTupleInferenceInstance(DeepQaTestCase):
    def setUp(self):
        super(TestTupleInferenceInstance, self).setUp()
        q_idx = 0
        answer_options = ("cat<>is<>a mammal<>context: which is nocturnal mammal$$$cat<>is<>nocturnal<>which " +
                          "is a nocturnal mammal###snake<>is<>a mammal<>which is a nocturnal " +
                          "mammal$$$snake<>is<>nocturnal<>which is a nocturnal mammal")
        background = ("cat<>eat<>food$$$cat<>is<>a mammal<>with fur<>which is a mammal$$$snake<>is example " +
                      "of<>reptile$$$some snakes<>are<>highly venomous")
        label = "1"
        self.line = "\t".join(str(x) for x in [q_idx, answer_options, background, label])
        self.instance = TupleInferenceInstance.read_from_line(self.line)

        q_idx2 = 1
        answer_options2 = ("erosion<>cause<>formation$$$erosion<>cause<>grandcanyon<>what caused formation of " +
                           "grand canyon######earthquakes<>cause<>formation$$$earthquakes<>cause<>grandcanyon" +
                           "<>what caused the formation of the grand canyon")
        background2 = ("erosion<>leads to<>breaking down rocks<>into smaller pieces$$$erosion<>create<>certain " +
                       "landforms$$$earthquakes<>occur<>frequently<>along fault lines")
        label2 = "2"
        line2 = "\t".join(str(x) for x in [q_idx2, answer_options2, background2, label2])
        self.instance_2 = TupleInferenceInstance.read_from_line(line2)

        q_idx3 = 2
        answer_options3 = ("solid water<>called<>water vapor<>what is solid water called###solid " +
                           "water<>called<>ice<>what is solid water called")
        background3 = ""
        label3 = "0"
        line3 = "\t".join(str(x) for x in [q_idx3, answer_options3, background3, label3])
        self.instance_3 = TupleInferenceInstance.read_from_line(line3)

        answer_options_simple = ("a<>sentence<>sentence<>context: a###a<>sentence<>sentence<>context: a")
        background_simple = ("a<>sentence<>sentence<>context: a$$$a<>sentence<>sentence<>context: a")
        label = "1"
        self.line_simple = "\t".join(str(x) for x in [q_idx, answer_options_simple, background_simple, label])


        # For testing indexed instances
        a1t1 = [[1, 2], [3]]
        a1t2 = [[1, 4, 5, 6], [1, 2, 3], [5, 6, 7]]
        answer_1 = [a1t1, a1t2]
        a2t1 = [[10, 3, 8], [1, 2]]
        answer_2 = [a2t1]
        a3t1 = [[1, 2], [3], [8, 9], [10]]
        a3t2 = [[10, 3, 8], [1, 2], [8]]
        a3t3 = [[10, 3, 8], [1, 2, 4]]
        a3t4 = [[10, 3, 8], [1, 2], [10, 1, 2], [2, 3], [2, 6]]
        answer_3 = [a3t1, a3t2, a3t3, a3t4]

        bt1 = [[10], [1], [8]]
        bt2 = [[10, 3, 8], [1, 2], [8], [10, 2, 5, 7, 10]]
        background_all = [bt1, bt2]
        self.indexed_instance = IndexedTupleInferenceInstance([answer_1, answer_2, answer_3],
                                                              background_all, 2, 0)

    def test_load_from_file_splits_correctly(self):
        # test general case
        assert len(self.instance.answer_tuples) == 2
        assert len(self.instance.answer_tuples[0]) == len(self.instance.answer_tuples[1]) == 2
        assert len(self.instance.background_tuples) == 4
        assert self.instance.label == 1
        assert len(self.instance.background_tuples[1].objects) == 3
        assert self.instance.background_tuples[1].subject == 'cat'
        assert self.instance.background_tuples[1].verb == 'is'
        # test no tuples for answer option
        assert len(self.instance_2.answer_tuples) == 3
        assert len(self.instance_2.answer_tuples[1]) == 0
        # test no background tuples
        assert len(self.instance_3.background_tuples) == 0

    def test_words_method(self):
        words = self.instance.words()['words']
        word_set = set(words)
        expected_words = set(("cat is a mammal which is nocturnal mammal cat is nocturnal which is a nocturnal " +
                              "mammal snake is a mammal which is a nocturnal mammal snake is nocturnal which is " +
                              "a nocturnal mammal cat eat food cat is a mammal with fur which is a mammal snake " +
                              "is example of reptile some snakes are highly venomous").split(" "))
        assert word_set == expected_words

    def test_indexed_instance_padding(self):
        data_indexer = DataIndexer()
        dataset = TextDataset([self.instance])
        data_indexer.fit_word_dictionary(dataset)

        indexed = self.instance.to_indexed_instance(data_indexer)
        num_question_tuples = 1
        num_background_tuples = 4
        num_slots = 3
        slot_length = 6
        num_options = 4
        padding_lengths = {'num_question_tuples': num_question_tuples,
                           'num_background_tuples': num_background_tuples,
                           'num_slots': num_slots,
                           'num_sentence_words': slot_length,
                           'num_options': num_options}
        indexed.pad(padding_lengths)
        assert len(indexed.answers_indexed) == num_options
        for answer_option_tuples in indexed.answers_indexed:
            assert len(answer_option_tuples) == num_question_tuples
            for ans_tuple in answer_option_tuples:
                assert len(ans_tuple) == num_slots
                for slot in ans_tuple:
                    assert len(slot) == slot_length
        assert len(indexed.background_indexed) == num_background_tuples
        for background_tuple in indexed.background_indexed:
            assert len(background_tuple) == num_slots
            for slot in background_tuple:
                assert len(slot) == slot_length

    def test_as_training_data_produces_correct_numpy_arrays(self):
        padding_lengths = {'num_question_tuples': 2,
                           'num_background_tuples': 3,
                           'num_slots': 2,
                           'num_sentence_words': 2,
                           'num_options': 3}
        self.indexed_instance.pad(padding_lengths)

        inputs, label = self.indexed_instance.as_training_data()
        assert numpy.all(label == numpy.asarray([0, 0, 1]))

        desired_options = numpy.asarray([[[[1, 2], [0, 3]],
                                          [[5, 6], [2, 3]]],
                                         [[[3, 8], [1, 2]],
                                          [[0, 0], [0, 0]]],
                                         [[[1, 2], [0, 3]],
                                          [[3, 8], [1, 2]]]], dtype='int32')

        desired_background = numpy.asarray([[[0, 10], [0, 1]],
                                            [[3, 8], [1, 2]],
                                            [[0, 0], [0, 0]]], dtype='int32')

        assert numpy.all([inputs[0]] == desired_options)
        assert numpy.all([inputs[1]] == desired_background)

    def test_works_with_word_and_character_tokenizer(self):
        answer_options_simple = ("a<>a sentence<><>")
        background_simple = ("a<>a sentence<><>")
        line_simple = "\t".join(str(x) for x in [answer_options_simple, background_simple, "0"])
        TextInstance.tokenizer = WordAndCharacterTokenizer(Params({}))
        data_indexer = DataIndexer()
        a_word_index = data_indexer.add_word_to_index("a", namespace='words')
        sentence_index = data_indexer.add_word_to_index("sentence", namespace='words')
        a_index = data_indexer.add_word_to_index("a", namespace='characters')
        s_index = data_indexer.add_word_to_index("s", namespace='characters')
        e_index = data_indexer.add_word_to_index("e", namespace='characters')

        new_instance = TupleInferenceInstance.read_from_line(line_simple)
        indexed = new_instance.to_indexed_instance(data_indexer)

        padding_lengths = {'num_question_tuples': 1,
                           'num_background_tuples': 1,
                           'num_slots': 2,
                           'num_sentence_words': 2,
                           'num_options': 1,
                           'num_word_characters': 3}
        indexed.pad(padding_lengths)
        expected_indexed_tuple = [[[0, 0, 0], [a_word_index, a_index, 0]],
                                  [[a_word_index, a_index, 0], [sentence_index, s_index, e_index]]]
        expected_answers_indexed = numpy.asarray([expected_indexed_tuple])
        expected_background_indexed = numpy.asarray(expected_indexed_tuple)
        assert numpy.all(indexed.answers_indexed == expected_answers_indexed)
        assert numpy.all(indexed.background_indexed == expected_background_indexed)
        TextInstance.tokenizer = tokenizers['words'](Params({}))
