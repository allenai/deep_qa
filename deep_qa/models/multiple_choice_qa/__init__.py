from .question_answer_similarity import QuestionAnswerSimilarity
from .tuple_entailment import MultipleChoiceTupleEntailmentModel
from .tuple_inference import TupleInferenceModel

concrete_models = {  # pylint: disable=invalid-name
        'QuestionAnswerSimilarity': QuestionAnswerSimilarity,
        'MultipleChoiceTupleEntailmentModel': MultipleChoiceTupleEntailmentModel,
        'TupleInferenceModel': TupleInferenceModel,
        }
