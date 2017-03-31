# pylint: disable=no-self-use,invalid-name

import numpy
from keras.layers import Input, Embedding
from keras.models import Model

from deep_qa.layers.entailment_models import DecomposableAttentionEntailment

class TestDecomposableAttention:
    def test_decomposable_attention_does_not_crash(self):
        sentence_length = 5
        embedding_dim = 10
        vocabulary_size = 15
        num_sentences = 7
        premise_input_layer = Input(shape=(sentence_length,), dtype='int32')
        hypothesis_input_layer = Input(shape=(sentence_length,), dtype='int32')
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, mask_zero=True)
        embedded_premise = embedding(premise_input_layer)
        embedded_hypothesis = embedding(hypothesis_input_layer)
        entailment_layer = DecomposableAttentionEntailment()
        entailment_scores = entailment_layer([embedded_premise, embedded_hypothesis])
        model = Model(inputs=[premise_input_layer, hypothesis_input_layer], outputs=entailment_scores)
        premise_input = numpy.random.randint(0, vocabulary_size, (num_sentences, sentence_length))
        hypothesis_input = numpy.random.randint(0, vocabulary_size, (num_sentences, sentence_length))
        model.predict([premise_input, hypothesis_input])
