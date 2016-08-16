from .nn_solver import NNSolver
from .lstm_solver import LSTMSolver
from .tree_lstm_solver import TreeLSTMSolver
from .memory_network import MemoryNetworkSolver

concrete_solvers = {
        'LSTMSolver': LSTMSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        }
