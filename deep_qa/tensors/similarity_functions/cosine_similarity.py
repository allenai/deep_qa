from typing import List, Tuple

from keras import backend as K

from .similarity_function import SimilarityFunction


class CosineSimilarity(SimilarityFunction):
    """
    This similarity function simply computes the cosine similarity between each pair of vectors.  It has
    no parameters.
    """
    def __init__(self, **kwargs):
        super(CosineSimilarity, self).__init__(**kwargs)

    def initialize_weights(self, input_shape: Tuple[int]) -> List['K.variable']:
        return []

    def compute_similarity(self, tensor_1, tensor_2):
        return K.sum(K.l2_normalize(tensor_1, axis=-1) * K.l2_normalize(tensor_2, axis=-1),
                     axis=-1)
