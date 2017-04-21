# pylint: disable=no-self-use,invalid-name
import numpy

from deep_qa.models.sequence_tagging import SimpleTagger
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestSimpleTagger(DeepQaTestCase):
    def test_trains_and_loads_correctly(self):
        self.write_sequence_tagging_files()
        args = Params({
                'save_models': True,
                'show_summary_with_masking_info': True,
                'instance_type': 'PreTokenizedTaggingInstance',
                'tokenizer': {'processor': {'word_splitter': 'no_op'}},
                })
        self.ensure_model_trains_and_loads(SimpleTagger, args)

    def test_loss_function_uses_mask(self):
        # We're going to make sure that the loss and accuracy computations are the same for any
        # permutation of labels on padded tokens.  If not, the loss/accuracy function is paying
        # attention to the labels when it shouldn't be.  We're not going to test for any particular
        # accuracy value, just that all of them are the same - I ran this a few times by hand to be
        # sure that we're getting different accuracy values, depending on the initialization.
        self.write_sequence_tagging_files()
        args = Params({
                'show_summary_with_masking_info': True,
                'instance_type': 'PreTokenizedTaggingInstance',
                'tokenizer': {'processor': {'word_splitter': 'no_op'}},
                })
        model = self.get_model(SimpleTagger, args)
        model.train()

        input_indices = [3, 2, 0, 0]
        labels = [[[0, 1], [1, 0], [1, 0], [1, 0]],
                  [[0, 1], [1, 0], [1, 0], [0, 1]],
                  [[0, 1], [1, 0], [0, 1], [1, 0]],
                  [[0, 1], [1, 0], [0, 1], [0, 1]]]
        results = [model.model.evaluate(numpy.asarray([input_indices]), numpy.asarray([label]))
                   for label in labels]
        loss, accuracy = zip(*results)
        assert len(set(loss)) == 1
        assert len(set(accuracy)) == 1
