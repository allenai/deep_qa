from .differentiable_search import DifferentiableSearchSolver
from .lstm_solver import LSTMSolver
from .memory_network import MemoryNetworkSolver
from .multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver
from .multiple_choice_similarity import MultipleChoiceSimilaritySolver
from .question_answer_lstm_solver import QuestionAnswerLSTMSolver
from .question_answer_memory_network import QuestionAnswerMemoryNetworkSolver
from .tree_lstm_solver import TreeLSTMSolver

from ..training import concrete_pretrainers
from .pretraining.attention_pretrainer import AttentionPretrainer
from .pretraining.snli_pretrainer import SnliAttentionPretrainer, SnliEntailmentPretrainer
from .pretraining.encoder_pretrainer import EncoderPretrainer

concrete_solvers = {  # pylint: disable=invalid-name
        'LSTMSolver': LSTMSolver,
        'TreeLSTMSolver': TreeLSTMSolver,
        'MemoryNetworkSolver': MemoryNetworkSolver,
        'DifferentiableSearchSolver': DifferentiableSearchSolver,
        'MultipleChoiceMemoryNetworkSolver': MultipleChoiceMemoryNetworkSolver,
        'QuestionAnswerMemoryNetworkSolver': QuestionAnswerMemoryNetworkSolver,
        'MultipleChoiceSimilaritySolver': MultipleChoiceSimilaritySolver,
        'QuestionAnswerLSTMSolver': QuestionAnswerLSTMSolver,
        }

concrete_pretrainers['AttentionPretrainer'] = AttentionPretrainer
concrete_pretrainers['SnliAttentionPretrainer'] = SnliAttentionPretrainer
concrete_pretrainers['SnliEntailmentPretrainer'] = SnliEntailmentPretrainer
concrete_pretrainers['EncoderPretrainer'] = EncoderPretrainer
