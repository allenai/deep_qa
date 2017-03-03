# pylint: disable=no-self-use,invalid-name
from deep_qa.models.multiple_choice_qa.question_answer_memory_network import QuestionAnswerMemoryNetwork
from ...common.test_case import DeepQaTestCase


class TestQuestionAnswerMemoryNetwork(DeepQaTestCase):
    def test_train_does_not_crash(self):
        self.write_question_answer_memory_network_files()
        model = self.get_model(QuestionAnswerMemoryNetwork)
        model.train()
