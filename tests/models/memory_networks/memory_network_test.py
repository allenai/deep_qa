# pylint: disable=no-self-use,invalid-name

from deep_qa.models.memory_networks import MemoryNetwork
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestMemoryNetwork(DeepQaTestCase):
    def setUp(self):
        super(TestMemoryNetwork, self).setUp()
        self.write_memory_network_files()

    def test_train_does_not_crash_with_defaults(self):
        model = self.get_model(MemoryNetwork)
        model.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = Params({'knowledge_selector': {'type': 'parameterized'}})
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_parameterized_heuristic_matching_knowledge_selector(self):
        args = Params({'knowledge_selector': {'type': 'parameterized_heuristic_matching'}})
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_cnn_encoder(self):
        args = Params({
                'num_sentence_words': 5,
                'encoder': {'default': {'type': 'cnn',
                                        'ngram_filter_sizes': [2, 3],
                                        'num_filters': 5}}
                })
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_positional_encoder(self):
        args = Params({'encoder': {'default': {'type': 'positional'}}})
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_attentive_gru_knowledge_combiner(self):
        args = Params({'knowledge_combiner': {'type': 'attentive_gru'}})
        model = self.get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_fusion_layer(self):
        args = Params({'knowledge_encoder': {'type': 'bi_gru'}})
        model = self.get_model(MemoryNetwork, args)
        model.train()
