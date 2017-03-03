# pylint: disable=no-self-use,invalid-name
from unittest import mock

from deep_qa.models.multiple_choice_qa.question_answer_similarity import QuestionAnswerSimilarity
from ...common.test_case import DeepQaTestCase


class TestQuestionAnswerSimilarity(DeepQaTestCase):
    def test_train_does_not_crash(self):
        self.write_question_answer_memory_network_files()
        model = self.get_model(QuestionAnswerSimilarity)
        model.train()

    @mock.patch.object(QuestionAnswerSimilarity, '_output_debug_info')
    def test_padding_works_correctly(self, _output_debug_info):
        self.write_question_answer_memory_network_files()
        args = {
                'num_hidden_layers': 1,
                'hidden_layer_width': 2,
                'show_summary_with_masking_info': True,
                'debug': {
                        'data': 'training',
                        'layer_names': ['answer_similarity_softmax']
                        }
                }
        model = self.get_model(QuestionAnswerSimilarity, args)

        def new_debug(output_dict, epoch):  # pylint: disable=unused-argument
            # We're going to check in here that the attentions and so on are properly masked.  In
            # particular, we'll check two things: (1) that the final answer option softmax has
            # correctly padded out the extra option, and (2) that the attention weights on all of
            # the inputs are properly padded.  To see that this test is correct, you have to look
            # at the actual file written in `write_multiple_true_false_memory_network_files()`.
            print(output_dict)
            answer_scores = output_dict['answer_similarity_softmax']
            assert answer_scores[0][2] == 0
            assert answer_scores[1][2] == 0
            assert answer_scores[3][2] == 0
        _output_debug_info.side_effect = new_debug
        model.train()
