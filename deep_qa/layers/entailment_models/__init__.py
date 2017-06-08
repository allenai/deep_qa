from .decomposable_attention import DecomposableAttentionEntailment
from .multiple_choice_tuple_entailment import MultipleChoiceTupleEntailment

entailment_models = { # pylint: disable=invalid-name
        'decomposable_attention': DecomposableAttentionEntailment,
        'multiple_choice_tuple_attention': MultipleChoiceTupleEntailment,
        }
