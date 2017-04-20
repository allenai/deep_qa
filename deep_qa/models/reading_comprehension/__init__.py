from .attention_sum_reader import AttentionSumReader
from .bidirectional_attention import BidirectionalAttentionFlow
from .gated_attention_reader import GatedAttentionReader

concrete_models = {  # pylint: disable=invalid-name
        'AttentionSumReader': AttentionSumReader,
        'BidirectionalAttentionFlow': BidirectionalAttentionFlow,
        'GatedAttentionReader': GatedAttentionReader,
        }
