from .differentiable_search import DifferentiableSearchMemoryNetwork
from .memory_network import MemoryNetwork
from .softmax_memory_network import SoftmaxMemoryNetwork

concrete_models = {  # pylint: disable=invalid-name
        'DifferentiableSearchMemoryNetwork': DifferentiableSearchMemoryNetwork,
        'MemoryNetwork': MemoryNetwork,
        'SoftmaxMemoryNetwork': SoftmaxMemoryNetwork,
        }
