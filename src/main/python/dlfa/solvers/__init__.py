from .nn_solver import NNSolver
from .lstm_solver import LSTMSolver
from .tree_lstm_solver import TreeLSTMSolver
from .memory_network import MemoryNetworkSolver
from .differentiable_search import DifferentiableSearchSolver
from .multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from .question_answer_memory_network import QuestionAnswerMemoryNetworkSolver

concrete_solvers = {  # pylint: disable=invalid-name
        'LSTMSolver': LSTMSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        'DifferentiableSearchSolver': DifferentiableSearchSolver,
        'MultipleChoiceMemoryNetworkSolver': MultipleChoiceMemoryNetworkSolver,
        'QuestionAnswerMemoryNetworkSolver': QuestionAnswerMemoryNetworkSolver,
        }
