# pylint: disable=no-self-use,invalid-name

from deep_qa.models.memory_networks.pretrainers.snli_pretrainer import SnliEntailmentPretrainer
from deep_qa.models.memory_networks.pretrainers.snli_pretrainer import SnliAttentionPretrainer
from deep_qa.models.memory_networks.memory_network import MemoryNetwork
from deep_qa.models.multiple_choice_qa.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetwork
from deep_qa.models.multiple_choice_qa.question_answer_memory_network import QuestionAnswerMemoryNetwork
from ....common.test_case import DeepQaTestCase


class TestSnliPretrainers(DeepQaTestCase):
    # pylint: disable=protected-access

    def setUp(self):
        super(TestSnliPretrainers, self).setUp()
        self.write_snli_file()
        self.write_memory_network_files()
        self.pretrainer_params = {"train_files": [self.SNLI_FILE]}

    def test_entailment_pretraining_does_not_crash_with_memory_network(self):
        model = self.get_model(MemoryNetwork)
        pretrainer = SnliEntailmentPretrainer(model, self.pretrainer_params)
        pretrainer.train()

    def test_entailment_pretraining_does_not_crash_with_multiple_true_false_memory_network(self):
        model = self.get_model(MultipleTrueFalseMemoryNetwork)
        pretrainer = SnliEntailmentPretrainer(model, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_memory_network(self):
        model = self.get_model(MemoryNetwork)
        pretrainer = SnliAttentionPretrainer(model, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_multiple_true_false_memory_network(self):
        model = self.get_model(MultipleTrueFalseMemoryNetwork)
        pretrainer = SnliAttentionPretrainer(model, self.pretrainer_params)
        pretrainer.train()

    def test_attention_pretraining_does_not_crash_with_question_answer_memory_network(self):
        model = self.get_model(QuestionAnswerMemoryNetwork)
        pretrainer = SnliAttentionPretrainer(model, self.pretrainer_params)
        pretrainer.train()

    def test_model_training_works_after_pretraining(self):
        # TODO(matt): It's possible in this test that the pretrainers don't actually get called,
        # and we wouldn't know it.  Not sure how to make sure that the pretrainers are actually
        # called in this test.  You could probably do it with some complicated use of patching...
        args = {
                'pretrainers': [
                        {
                                'type': 'SnliEntailmentPretrainer',
                                'num_epochs': 1,
                                'train_files': [self.SNLI_FILE]
                                },
                        {
                                'type': 'SnliAttentionPretrainer',
                                'num_epochs': 1,
                                'train_files': [self.SNLI_FILE]
                                },
                        ]
                }
        model = self.get_model(MemoryNetwork, args)
        model.train()
