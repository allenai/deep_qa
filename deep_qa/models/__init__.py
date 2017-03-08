from .entailment.decomposable_attention import DecomposableAttention
from .memory_networks.differentiable_search import DifferentiableSearchMemoryNetwork
from .memory_networks.memory_network import MemoryNetwork
from .memory_networks.softmax_memory_network import SoftmaxMemoryNetwork
from .multiple_choice_qa.decomposable_attention import MultipleTrueFalseDecomposableAttention
from .multiple_choice_qa.tuple_inference import TupleInferenceModel
from .multiple_choice_qa.multiple_true_false_memory_network import MultipleTrueFalseMemoryNetwork
from .multiple_choice_qa.multiple_true_false_similarity import MultipleTrueFalseSimilarity
from .multiple_choice_qa.question_answer_memory_network import QuestionAnswerMemoryNetwork
from .multiple_choice_qa.question_answer_similarity import QuestionAnswerSimilarity
from .multiple_choice_qa.tuple_entailment import MultipleChoiceTupleEntailmentModel
from .sequence_tagging.simple_tagger import SimpleTagger
from .reading_comprehension.attention_sum_reader import AttentionSumReader
from .reading_comprehension.gated_attention_reader import GatedAttentionReader
from .reading_comprehension.bidirectional_attention import BidirectionalAttentionFlow
from .text_classification.tree_lstm_model import TreeLSTMModel
from .text_classification.true_false_model import TrueFalseModel

from ..training import concrete_pretrainers
from .memory_networks.pretrainers.attention_pretrainer import AttentionPretrainer
from .memory_networks.pretrainers.snli_pretrainer import SnliAttentionPretrainer, SnliEntailmentPretrainer
from .text_pretrainers.encoder_pretrainer import EncoderPretrainer

concrete_models = {  # pylint: disable=invalid-name
        'AttentionSumReader': AttentionSumReader,
        'BidirectionalAttentionFlow': BidirectionalAttentionFlow,
        'DecomposableAttention': DecomposableAttention,
        'DifferentiableSearchMemoryNetwork': DifferentiableSearchMemoryNetwork,
        'GatedAttentionReader': GatedAttentionReader,
        'MemoryNetwork': MemoryNetwork,
        'MultipleTrueFalseDecomposableAttention': MultipleTrueFalseDecomposableAttention,
        'MultipleTrueFalseMemoryNetwork': MultipleTrueFalseMemoryNetwork,
        'MultipleTrueFalseSimilarity': MultipleTrueFalseSimilarity,
        'MultipleChoiceTupleEntailmentModel': MultipleChoiceTupleEntailmentModel,
        'QuestionAnswerMemoryNetwork': QuestionAnswerMemoryNetwork,
        'QuestionAnswerSimilarity': QuestionAnswerSimilarity,
        'SimpleTagger': SimpleTagger,
        'SoftmaxMemoryNetwork': SoftmaxMemoryNetwork,
        'TreeLSTMModel': TreeLSTMModel,
        'TrueFalseModel': TrueFalseModel,
        'TupleInferenceModel': TupleInferenceModel
}

concrete_pretrainers['AttentionPretrainer'] = AttentionPretrainer
concrete_pretrainers['SnliAttentionPretrainer'] = SnliAttentionPretrainer
concrete_pretrainers['SnliEntailmentPretrainer'] = SnliEntailmentPretrainer
concrete_pretrainers['EncoderPretrainer'] = EncoderPretrainer
