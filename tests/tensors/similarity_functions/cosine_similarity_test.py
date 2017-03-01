# pylint: disable=no-self-use,invalid-name

import numpy
from numpy.testing import assert_almost_equal
import keras.backend as K

from deep_qa.tensors.similarity_functions.cosine_similarity import CosineSimilarity
from deep_qa.tensors.similarity_functions.dot_product import DotProduct

class TestCosineSimilarityFunction:
    cosine_similarity = CosineSimilarity(name='cosine_similarity')
    dot_product = DotProduct(name="dot_product")

    def test_initialize_weights_returns_empty(self):
        weights = self.cosine_similarity.initialize_weights(input_shape=(2, 3))
        assert isinstance(weights, list) and len(weights) == 0

    def test_compute_similarity_does_a_cosine_similarity(self):
        a_vectors = numpy.asarray([[numpy.random.random(3) for _ in range(2)]], dtype="float32")
        b_vectors = numpy.asarray([[numpy.random.random(3) for _ in range(2)]], dtype="float32")
        normed_a = K.l2_normalize(K.variable(a_vectors), axis=-1)
        normed_b = K.l2_normalize(K.variable(b_vectors), axis=-1)
        desired_result = K.eval(self.dot_product.compute_similarity(normed_a, normed_b))
        result = K.eval(self.cosine_similarity.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (1, 2)    # batch_size = 1
        assert numpy.all(result == desired_result)

    def test_compute_similarity_works_with_higher_order_tensors(self):
        a_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        b_vectors = numpy.random.rand(5, 4, 3, 6, 7)
        normed_a = K.eval(K.l2_normalize(K.variable(a_vectors), axis=-1))
        normed_b = K.eval(K.l2_normalize(K.variable(b_vectors), axis=-1))
        result = K.eval(self.cosine_similarity.compute_similarity(K.variable(a_vectors), K.variable(b_vectors)))
        assert result.shape == (5, 4, 3, 6)
        assert_almost_equal(result[3, 2, 1, 3],
                            numpy.dot(normed_a[3, 2, 1, 3], normed_b[3, 2, 1, 3]),
                            decimal=6)
