from .babi_instance import BabiInstance
from .background_instance import BackgroundInstance
from .logical_form_instance import LogicalFormInstance
from .multiple_true_false_instance import MultipleTrueFalseInstance
from .question_answer_instance import QuestionAnswerInstance
from .sentence_pair_instance import SentencePairInstance
from .snli_instance import SnliInstance
from .true_false_instance import TrueFalseInstance

instances = {  # pylint: disable=invalid-name
        'BabiInstance': BabiInstance,
        'BackgroundInstance': BackgroundInstance,
        'LogicalFormInstance': LogicalFormInstance,
        'MultipleTrueFalseInstance': MultipleTrueFalseInstance,
        'QuestionAnswerInstance': QuestionAnswerInstance,
        'SentencePairInstance': SentencePairInstance,
        'SnliInstance': SnliInstance,
        'TrueFalseInstance': TrueFalseInstance,
        }
