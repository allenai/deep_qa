from .encoded_sentence import TrueFalseEntailmentModel, MultipleChoiceEntailmentModel
from .encoded_sentence import QuestionAnswerEntailmentModel, MemoryOnlyCombiner, HeuristicMatchingCombiner
from .decomposable_attention import DecomposableAttentionEntailment
from .multiple_choice_tuple_entailment import MultipleChoiceTupleEntailment

entailment_models = { # pylint: disable=invalid-name
        'true_false_mlp': TrueFalseEntailmentModel,
        'multiple_choice_mlp': MultipleChoiceEntailmentModel,
        'question_answer_mlp': QuestionAnswerEntailmentModel,
        'decomposable_attention': DecomposableAttentionEntailment,
        'multiple_choice_tuple_attention': MultipleChoiceTupleEntailment,
        }

entailment_input_combiners = { # pylint: disable=invalid-name
        'memory_only': MemoryOnlyCombiner,
        'heuristic_matching': HeuristicMatchingCombiner,
        }
