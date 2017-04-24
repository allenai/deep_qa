# pylint: disable=no-self-use,invalid-name
from unittest import mock

from deep_qa.models.multiple_choice_qa import MultipleTrueFalseMemoryNetwork
from deep_qa.common.params import Params
from ...common.test_case import DeepQaTestCase


class TestMultipleTrueFalseMemoryNetwork(DeepQaTestCase):

    def setUp(self):
        super(TestMultipleTrueFalseMemoryNetwork, self).setUp()
        self.write_multiple_true_false_memory_network_files()

    def test_model_trains_and_loads_correctly(self):
        self.ensure_model_trains_and_loads(MultipleTrueFalseMemoryNetwork, {})

    @mock.patch.object(MultipleTrueFalseMemoryNetwork, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        args = Params({
                'num_options': 5,
                'embedding_dim': {"words": 2},
                'max_knowledge_length': 3,
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': [
                                'answer_option_softmax',
                                'timedist_knowledge_selector_0',
                                ],
                        }
                })
        model = self.get_model(MultipleTrueFalseMemoryNetwork, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check in here that the attentions and so on are properly masked.  In
            # particular, we'll check two things: (1) that the final answer option softmax has
            # correctly padded out the extra option, and (2) that the attention weights on all of
            # the inputs are properly padded.  To see that this test is correct, you have to look
            # at the actual file written in `write_multiple_true_false_memory_network_files()`.
            answer_scores = output_dict['answer_option_softmax'][0]
            assert answer_scores[4] == 0
            attention_weights = output_dict['timedist_knowledge_selector_0'][0]
            assert attention_weights[0][2] == 0
            assert attention_weights[1][1] == 0
            assert attention_weights[1][2] == 0
            assert attention_weights[2][1] == 0
            assert attention_weights[2][2] == 0
            assert attention_weights[3][2] == 0
            assert attention_weights[4][0] == 0
            assert attention_weights[4][1] == 0
            assert attention_weights[4][2] == 0
        _output_debug_info.side_effect = new_debug
        model.train()

    def test_train_does_not_crash_using_adaptive_recurrence(self):
        args = Params({'recurrence_mode': {'type': 'adaptive'}})
        model = self.get_model(MultipleTrueFalseMemoryNetwork, args)
        model.train()
