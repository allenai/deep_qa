from .pretokenized_tagging_instance import PreTokenizedTaggingInstance
from .tagging_instance import TaggingInstance, IndexedTaggingInstance

concrete_instances = {  # pylint: disable=invalid-name
        'PreTokenizedTaggingInstance': PreTokenizedTaggingInstance
        }
