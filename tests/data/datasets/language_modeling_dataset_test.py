# pylint: disable=no-self-use,invalid-name
from deep_qa.data.datasets import LanguageModelingDataset
from deep_qa.data.instances.language_modeling.sentence_instance import SentenceInstance
from deep_qa.common.params import Params
from tests.common.test_case import DeepQaTestCase


class TestLanguageModellingDataset(DeepQaTestCase):

    def setUp(self):
        super(TestLanguageModellingDataset, self).setUp()
        self.write_sentence_data()

    def test_read_from_file(self):
        args = Params({"sequence_length": 4})
        dataset = LanguageModelingDataset.read_from_file(self.TRAIN_FILE, SentenceInstance, args)

        instances = dataset.instances
        assert instances[0].text == "This is a sentence"
        assert instances[1].text == "for language modelling. Here's"
        assert instances[2].text == "another one for language"
