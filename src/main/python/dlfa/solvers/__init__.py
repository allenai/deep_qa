from .nn_solver import NNSolver
from .lstm_solver import LSTMSolver
from .tree_lstm_solver import TreeLSTMSolver
from .memory_network import MemoryNetworkSolver
from .differentiable_search import DifferentiableSearchSolver

concrete_solvers = {  # pylint: disable=invalid-name
        'LSTMSolver': LSTMSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        'DifferentiableSearchSolver': DifferentiableSearchSolver,
        }
