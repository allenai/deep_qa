# pylint: disable=no-self-use,invalid-name

from deep_qa.models.entailment import DecomposableAttention
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestDecomposableAttentionModel(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_snli_files()
        args = Params({
                'num_seq2seq_layers': 1,
                })
        self.ensure_model_trains_and_loads(DecomposableAttention, args)
