from .decomposable_attention import MultipleTrueFalseDecomposableAttention
from .multiple_true_false_memory_network import MultipleTrueFalseMemoryNetwork
from .multiple_true_false_similarity import MultipleTrueFalseSimilarity
from .question_answer_memory_network import QuestionAnswerMemoryNetwork
from .question_answer_similarity import QuestionAnswerSimilarity
from .tuple_entailment import MultipleChoiceTupleEntailmentModel
from .tuple_inference import TupleInferenceModel

concrete_models = {  # pylint: disable=invalid-name
        'MultipleTrueFalseDecomposableAttention': MultipleTrueFalseDecomposableAttention,
        'MultipleTrueFalseMemoryNetwork': MultipleTrueFalseMemoryNetwork,
        'MultipleTrueFalseSimilarity': MultipleTrueFalseSimilarity,
        'QuestionAnswerMemoryNetwork': QuestionAnswerMemoryNetwork,
        'QuestionAnswerSimilarity': QuestionAnswerSimilarity,
        'MultipleChoiceTupleEntailmentModel': MultipleChoiceTupleEntailmentModel,
        'TupleInferenceModel': TupleInferenceModel,
        }
