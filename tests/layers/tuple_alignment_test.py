# pylint: disable=no-self-use,invalid-name
import numpy
from keras.layers import Input
from keras.models import Model

from deep_qa.layers.entailment_models import MultipleChoiceTupleEntailment
from deep_qa.layers.time_distributed_embedding import TimeDistributedEmbedding

class TestTupleAlignment:
    def test_tuple_alignment_does_not_crash(self):
        question_length = 5
        num_options = 4
        tuple_size = 3
        num_tuples = 7
        embedding_dim = 10
        vocabulary_size = 15
        batch_size = 32
        question_input_layer = Input(shape=(question_length,), dtype='int32')
        answer_input_layer = Input(shape=(num_options,), dtype='int32')
        knowledge_input_layer = Input(shape=(num_tuples, tuple_size), dtype='int32')
        # Embedding does not mask zeros
        embedding = TimeDistributedEmbedding(input_dim=vocabulary_size, output_dim=embedding_dim,
                                             mask_zero=True)
        embedded_question = embedding(question_input_layer)
        embedded_answer = embedding(answer_input_layer)
        embedded_knowledge = embedding(knowledge_input_layer)
        entailment_layer = MultipleChoiceTupleEntailment({})
        entailment_scores = entailment_layer([embedded_knowledge, embedded_question, embedded_answer])
        model = Model(input=[knowledge_input_layer, question_input_layer, answer_input_layer],
                      output=entailment_scores)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        knowledge_input = numpy.random.randint(0, vocabulary_size, (batch_size, num_tuples, tuple_size))
        question_input = numpy.random.randint(0, vocabulary_size, (batch_size, question_length))
        answer_input = numpy.random.randint(0, vocabulary_size, (batch_size, num_options))
        model.predict([knowledge_input, question_input, answer_input])
