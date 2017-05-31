from deep_qa.models.sequence_tagging import VerbSemanticsModel
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestVerbSemanticsModel(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_verb_semantics_files()
        args = Params({
                'save_models': True,
                'show_summary_with_masking_info': True,
                'instance_type': 'VerbSemanticsInstance',
                'tokenizer': {'processor': {'word_splitter': 'no_op'}},
                # Since VerbSemanticsModel has 2 outputs, validation metric cannot be "validation accuracy",
                # hence setting it to "validation loss".
                'validation_metric': 'val_loss',
                })

        self.ensure_model_trains_and_loads(VerbSemanticsModel, args)
