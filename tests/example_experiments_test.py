# pylint: disable=invalid-name,no-self-use
import pyhocon

from deep_qa.common.params import Params, replace_none
from deep_qa.models import concrete_models

from deep_qa.testing.test_case import DeepQaTestCase


class TestExampleExperiments(DeepQaTestCase):
    def setUp(self):
        super(TestExampleExperiments, self).setUp()
        self.write_pretrained_vector_files()
        self.example_experiments_dir = "./example_experiments"
        self.entailment_dir = self.example_experiments_dir + "/entailment/"
        self.memory_networks_dir = self.example_experiments_dir + "/memory_networks/"
        self.multiple_choice_qa_dir = self.example_experiments_dir + "/multiple_choice_qa/"
        self.reading_comprehension_dir = self.example_experiments_dir + "/reading_comprehension/"
        self.sentence_selection_dir = self.example_experiments_dir + "/sentence_selection/"
        self.sequence_tagging_dir = self.example_experiments_dir + "/sequence_tagging/"

    def test_entailment_examples_can_train(self):
        self.write_snli_files()
        snli_decomposable_attention = self.entailment_dir + "snli_decomposable_attention.json"
        self.check_experiment_type_can_train(snli_decomposable_attention)

    def test_bidaf_can_train(self):
        self.write_span_prediction_files()
        bidaf_squad = self.reading_comprehension_dir + "bidaf_squad.json"
        self.check_experiment_type_can_train(bidaf_squad)

    def test_ga_reader_can_train(self):
        self.write_who_did_what_files()
        gareader_who_did_what = self.reading_comprehension_dir + "gareader_who_did_what.json"
        self.check_experiment_type_can_train(gareader_who_did_what)

    def test_as_reader_can_train(self):
        self.write_who_did_what_files()
        as_reader_who_did_what = self.reading_comprehension_dir + "asreader_who_did_what.json"
        self.check_experiment_type_can_train(as_reader_who_did_what)

    def test_simple_tagger_can_train(self):
        self.write_sequence_tagging_files()
        simple_tagger = self.sequence_tagging_dir + "simple_tagger.json"
        self.check_experiment_type_can_train(simple_tagger)

    def check_experiment_type_can_train(self, param_file):
        param_dict = pyhocon.ConfigFactory.parse_file(param_file)
        params = Params(replace_none(param_dict))
        model_class = concrete_models[params.pop("model_class")]
        # Tests will try to create root directories as we have /net/efs paths,
        # so we just remove the serialisation aspect here, alter the train/validation
        # paths to the dummy test ones and make sure we only do one epoch to
        # speed things up.
        params["model_serialization_prefix"] = None
        if len(params["train_files"]) > 1:
            params["train_files"] = [self.TRAIN_FILE, self.TRAIN_BACKGROUND]
            params["validation_files"] = [self.VALIDATION_FILE, self.VALIDATION_BACKGROUND]
        else:
            params["train_files"] = [self.TRAIN_FILE]
            params["validation_files"] = [self.TRAIN_FILE]
        params["num_epochs"] = 1
        try:
            if params["embeddings"]["words"]["pretrained_file"]:
                params["embeddings"]["words"]["pretrained_file"] = self.PRETRAINED_VECTORS_GZIP

        except KeyError:
            # No embedding/words field passed in the parameters,
            # so nothing to change.
            pass

        model = self.get_model(model_class, params)
        model.train()
