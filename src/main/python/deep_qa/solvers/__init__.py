from .no_memory.true_false_solver import TrueFalseSolver
from .no_memory.question_answer_solver import QuestionAnswerSolver
from .no_memory.tree_lstm_solver import TreeLSTMSolver
from .with_memory.differentiable_search import DifferentiableSearchSolver
from .with_memory.memory_network import MemoryNetworkSolver
from .with_memory.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetworkSolver
from .with_memory.multiple_true_false_similarity import MultipleTrueFalseSimilaritySolver
from .with_memory.question_answer_memory_network import QuestionAnswerMemoryNetworkSolver

from ..training import concrete_pretrainers
from .pretraining.attention_pretrainer import AttentionPretrainer
from .pretraining.snli_pretrainer import SnliAttentionPretrainer, SnliEntailmentPretrainer
from .pretraining.encoder_pretrainer import EncoderPretrainer

concrete_solvers = {  # pylint: disable=invalid-name
        'TrueFalseSolver': TrueFalseSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        'DifferentiableSearchSolver': DifferentiableSearchSolver,
        'MultipleTrueFalseMemoryNetworkSolver': MultipleTrueFalseMemoryNetworkSolver,
        'QuestionAnswerMemoryNetworkSolver': QuestionAnswerMemoryNetworkSolver,
        'MultipleTrueFalseSimilaritySolver': MultipleTrueFalseSimilaritySolver,
        'QuestionAnswerLSTMSolver': QuestionAnswerSolver,
        }

concrete_pretrainers['AttentionPretrainer'] = AttentionPretrainer
concrete_pretrainers['SnliAttentionPretrainer'] = SnliAttentionPretrainer
concrete_pretrainers['SnliEntailmentPretrainer'] = SnliEntailmentPretrainer
concrete_pretrainers['EncoderPretrainer'] = EncoderPretrainer
