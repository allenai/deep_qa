# pylint: disable=no-self-use,invalid-name

from unittest import TestCase
import os
import shutil

from deep_qa.models.memory_networks.memory_network import MemoryNetwork
from ...common.constants import TEST_DIR
from ...common.models import get_model
from ...common.models import write_memory_network_files


class TestMemoryNetwork(TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        os.makedirs(TEST_DIR, exist_ok=True)
        write_memory_network_files()

    def tearDown(self):
        shutil.rmtree(TEST_DIR)

    def test_train_does_not_crash_with_defaults(self):
        model = get_model(MemoryNetwork)
        model.train()

    def test_train_does_not_crash_with_parameterized_knowledge_selector(self):
        args = {'knowledge_selector': {'type': 'parameterized'}}
        model = get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_parameterized_heuristic_matching_knowledge_selector(self):
        args = {'knowledge_selector': {'type': 'parameterized_heuristic_matching'}}
        model = get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_cnn_encoder(self):
        args = {
                'max_sentence_length': 5,
                'encoder': {'type': 'cnn', 'ngram_filter_sizes': [2, 3], 'num_filters': 5}
                }
        model = get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_positional_encoder(self):
        args = {'encoder': {'type': 'positional'}}
        model = get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_attentive_gru_knowledge_combiner(self):
        args = {'knowledge_combiner': {'type': 'attentive_gru'}}
        model = get_model(MemoryNetwork, args)
        model.train()

    def test_train_does_not_crash_with_fusion_layer(self):
        args = {'knowledge_encoder': {'type': 'bi_gru'}}
        model = get_model(MemoryNetwork, args)
        model.train()
