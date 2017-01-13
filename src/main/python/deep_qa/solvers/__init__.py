from .no_memory.true_false_solver import TrueFalseSolver
from .no_memory.question_answer_solver import QuestionAnswerSolver
from .no_memory.tree_lstm_solver import TreeLSTMSolver
from .with_memory.decomposable_attention_solver import DecomposableAttentionSolver
from .with_memory.decomposable_attention_solver import MultipleTrueFalseDecomposableAttentionSolver
from .with_memory.differentiable_search import DifferentiableSearchSolver
from .with_memory.memory_network import MemoryNetworkSolver
from .with_memory.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetworkSolver
from .with_memory.multiple_true_false_similarity import MultipleTrueFalseSimilaritySolver
from .with_memory.softmax_memory_network import SoftmaxMemoryNetworkSolver
from .with_memory.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from .with_memory.tuple_entailment import MultipleChoiceTupleEntailmentSolver

from ..training import concrete_pretrainers
from .pretraining.attention_pretrainer import AttentionPretrainer
from .pretraining.snli_pretrainer import SnliAttentionPretrainer, SnliEntailmentPretrainer
from .pretraining.encoder_pretrainer import EncoderPretrainer

concrete_solvers = {  # pylint: disable=invalid-name
        'DecomposableAttentionSolver': DecomposableAttentionSolver,
        'DifferentiableSearchSolver': DifferentiableSearchSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        'MultipleTrueFalseDecomposableAttentionSolver': MultipleTrueFalseDecomposableAttentionSolver,
        'MultipleTrueFalseMemoryNetworkSolver': MultipleTrueFalseMemoryNetworkSolver,
        'MultipleTrueFalseSimilaritySolver': MultipleTrueFalseSimilaritySolver,
        'MultipleChoiceTupleEntailmentSolver': MultipleChoiceTupleEntailmentSolver,
        'QuestionAnswerMemoryNetworkSolver': QuestionAnswerMemoryNetworkSolver,
        'QuestionAnswerSolver': QuestionAnswerSolver,
        'SoftmaxMemoryNetworkSolver': SoftmaxMemoryNetworkSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'TrueFalseSolver': TrueFalseSolver,
        }

concrete_pretrainers['AttentionPretrainer'] = AttentionPretrainer
concrete_pretrainers['SnliAttentionPretrainer'] = SnliAttentionPretrainer
concrete_pretrainers['SnliEntailmentPretrainer'] = SnliEntailmentPretrainer
concrete_pretrainers['EncoderPretrainer'] = EncoderPretrainer
