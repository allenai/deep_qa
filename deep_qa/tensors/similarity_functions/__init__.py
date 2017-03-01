from collections import OrderedDict

from .bilinear import Bilinear
from .dot_product import DotProduct
from .linear import Linear
from .cosine_similarity import CosineSimilarity

# The first item added here will be used as the default in some cases.
similarity_functions = OrderedDict()  # pylint: disable=invalid-name
similarity_functions['dot_product'] = DotProduct
similarity_functions['bilinear'] = Bilinear
similarity_functions['linear'] = Linear
similarity_functions['cosine_similarity'] = CosineSimilarity
