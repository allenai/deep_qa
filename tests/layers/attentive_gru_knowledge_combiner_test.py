# pylint: disable=no-self-use
import numpy

from keras.layers import Input, Embedding, merge
from keras.models import Model
import keras.backend as K

from deep_qa.layers.knowledge_combiners import AttentiveGRUKnowledgeCombiner


class TestAttentiveGRUKnowledgeCombiner:
    def test_on_unmasked_input(self):

        sentence_length = 5
        embedding_size = 10
        vocabulary_size = 15
        input_layer = Input(shape=(sentence_length,), dtype='int32')
        attention = Input(shape=(sentence_length,), dtype='float32')
        # Embedding does not mask zeros
        embedding = Embedding(input_dim=vocabulary_size, output_dim=embedding_size)
        attentive_gru = AttentiveGRUKnowledgeCombiner(output_dim=embedding_size,
                                                      input_length=sentence_length,
                                                      return_sequences=True,
                                                      name='attentive_gru_test')
        embedded_input = embedding(input_layer)
        concat_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=2),
                                                        layer_outs[1]],
                                                       axis=2)

        combined_sentence_with_attention = merge([attention, embedded_input],
                                                 mode=concat_mode,
                                                 output_shape=(5, 11))

        sequence_of_outputs = attentive_gru(combined_sentence_with_attention)
        model = Model(input=[input_layer, attention], output=sequence_of_outputs)
        model.compile(loss="mse", optimizer="sgd")  # Will not train this model
        test_input = numpy.asarray([[0, 3, 1, 7, 10]], dtype='int32')
        attention_input = numpy.asarray([[1., 0., 0., 0., 0.]], dtype='float32')

        # To debug this model, we are going to check that if we pass an attention mask into
        # the attentive_gru which has all zeros apart from the first element which is one,
        # all the elements should be equal to the first output as the state won't change over
        # time, as we add in none of the memory. This is not the intended use of this class,
        # but if this works, the intended use will be correct.
        actual_sequence_of_outputs = numpy.squeeze(model.predict([test_input, attention_input]))
        for i in range(sentence_length - 1):
            assert numpy.array_equal(actual_sequence_of_outputs[i, :], actual_sequence_of_outputs[i+1, :])
