
import keras.backend as K

# Individual layers.
from .additive import Additive
from .bigru_index_selector import BiGRUIndexSelector
from .complex_concat import ComplexConcat
from .highway import Highway
from .l1_normalize import L1Normalize
from .masked_layer import MaskedLayer
from .noisy_or import BetweenZeroAndOne, NoisyOr
from .option_attention_sum import OptionAttentionSum
from .overlap import Overlap
from .time_distributed_embedding import TimeDistributedEmbedding
from .top_knowledge_selector import TopKnowledgeSelector
from .vector_matrix_merge import VectorMatrixMerge
from .vector_matrix_split import VectorMatrixSplit

# Groups of layers with their dictionaries.
from .knowledge_combiners import knowledge_combiners, WeightedAverageKnowledgeCombiner
from .knowledge_combiners import AttentiveGRUKnowledgeCombiner
from .knowledge_encoders import knowledge_encoders, IndependentKnowledgeEncoder
from .knowledge_encoders import TemporalKnowledgeEncoder, BiGRUKnowledgeEncoder
from .knowledge_selectors import selectors, DotProductKnowledgeSelector
from .knowledge_selectors import ParameterizedKnowledgeSelector
from .knowledge_selectors import ParameterizedHeuristicMatchingKnowledgeSelector
from .memory_updaters import updaters, DenseConcatMemoryUpdater
from .memory_updaters import DenseConcatNoQuestionMemoryUpdater, SumMemoryUpdater
from .recurrence_modes import recurrence_modes, FixedRecurrence
from .recurrence_modes import AdaptiveRecurrence
