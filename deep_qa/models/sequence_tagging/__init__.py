from .verb_semantics_model import VerbSemanticsModel
from .simple_tagger import SimpleTagger

concrete_models = {  # pylint: disable=invalid-name
        'SimpleTagger': SimpleTagger,
        'VerbSemanticsModel': VerbSemanticsModel,
        }
