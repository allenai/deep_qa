from .entailment import concrete_models as entailment_models
from .memory_networks import concrete_models as memory_network_models
from .multiple_choice_qa import concrete_models as multiple_choice_qa_models
from .sentence_selection import concrete_models as sentence_selection_models
from .sequence_tagging import concrete_models as sequence_tagging_models
from .reading_comprehension import concrete_models as reading_comprehension_models
from .text_classification import concrete_models as text_classification_models

concrete_models = {}  # pylint: disable=invalid-name
__concrete_task_models = [  # pylint: disable=invalid-name
        entailment_models,
        memory_network_models,
        multiple_choice_qa_models,
        sentence_selection_models,
        sequence_tagging_models,
        reading_comprehension_models,
        text_classification_models,
        ]
for models_for_task in __concrete_task_models:
    for model_name, model_class in models_for_task.items():
        if model_name in concrete_models:
            raise RuntimeError("Duplicate model name found: " + model_name)
        concrete_models[model_name] = model_class
