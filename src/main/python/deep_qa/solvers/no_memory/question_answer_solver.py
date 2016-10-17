from typing import Any, Dict
from overrides import overrides

from keras import backend as K
from keras.layers import Dense, Dropout, Lambda, TimeDistributed
from keras.models import Model

from ...data.instances.question_answer_instance import QuestionAnswerInstance
from ...training.text_trainer import TextTrainer


class QuestionAnswerSolver(TextTrainer):
    """
    A TextTrainer that takes a question and several answer options as input, encodes the word
    sequences using a sentence encoder, optionally passes the question encoding through some dense
    layers, then selects the option that is most similar to the final question encoding.

    This assumes that you can get the parameters of the model to learn whatever associations
    between words in the question and words in the answer are necessary to select the correct
    choice.  There is no notion of external memory or background knowledge here.
    """
    def __init__(self, params: Dict[str, Any]):
        self.num_hidden_layers = params.pop('num_hidden_layers', 1)
        self.hidden_layer_width = params.pop('hidden_layer_width', 50)
        self.hidden_layer_activation = params.pop('hidden_layer_activation', 'relu')
        self.max_answer_length = params.pop('max_answer_length', None)
        self.num_options = params.pop('num_options', None)
        super(QuestionAnswerSolver, self).__init__(params)

        self.answer_dim = self.embedding_size

        self.hidden_layers = []
        self.projection_layer = None
        self._init_layers()

    def _init_layers(self):
        self.hidden_layers = []
        for i in range(self.num_hidden_layers):
            self.hidden_layers.append(Dense(output_dim=self.hidden_layer_width,
                                            activation=self.hidden_layer_activation,
                                            name='question_hidden_layer_%d' % i))
        self.projection_layer = Dense(output_dim=self.answer_dim,
                                      activation='linear',
                                      name='question_projection')

    @staticmethod
    def _tile_projection(inputs):
        """
        This is used to do some fancy footwork in to compare a question encoding to a number of
        answer option encodings.
        """
        # We need to tile the projected_input so that we can easily do a dot product with the
        # encoded_answers.  This follows the logic in knowledge_selectors.tile_sentence_encoding.
        answers, projected = inputs
        # Shape: (num_options, batch_size, answer_dim)
        ones = K.permute_dimensions(K.ones_like(answers), [1, 0, 2])
        # Shape: (batch_size, num_options, answer_dim)
        return K.permute_dimensions(ones * projected, [1, 0, 2])

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass both questions and answers through an embedding
        layer, then through an encoder.  Then we'll pass the encoded question through a some dense
        layers, and compare the similar to the encoded answers.
        """
        # First we create input layers and pass the inputs through an embedding.
        question_input_layer, question_embedding = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,), name_prefix='sentence')
        answer_input_layer, answer_embedding = self._get_embedded_sentence_input(
                input_shape=(self.num_options, self.max_answer_length), name_prefix="answer")

        # Then we encode the question and answers using some encoder.
        question_encoder = self._get_sentence_encoder()
        encoded_question = question_encoder(question_embedding)
        # TODO(matt): make the dropout a parameter (should probably be "encoder_dropout", in
        # NNSolver).
        regularized_encoded_question = Dropout(0.2)(encoded_question)

        # This needs to use a new encoder because we can't compile the same LSTM with two different
        # padding lengths...  If you really want to use the same LSTM for both questions and
        # answers, pad the answers to the same dimension as the questions, replacing
        # self.max_answer_length with self.max_sentence_length everywhere.
        answer_encoder = TimeDistributed(question_encoder, name="answer_encoder")
        encoded_answers = answer_encoder(answer_embedding)

        # Then we pass the question through some hidden (dense) layers.
        hidden_input = regularized_encoded_question
        for layer in self.hidden_layers:
            hidden_input = layer(hidden_input)
        projected_input = self.projection_layer(hidden_input)

        # Lastly, we compare the similarity of the question to the answer options.

        # To make the similarity dot product more efficient, we tile the input first, so we can
        # just do an element-wise product and a sum.  Shape: (batch_size, num_options, answer_dim),
        # where the (batch_size, answer_dim) projected input has been replicated num_options times.
        # Note that these lambda layers have no parameters, so they don't need to be put into
        # self._init_layers().
        tile_layer = Lambda(self._tile_projection,
                            output_shape=lambda input_shapes: input_shapes[0],
                            name='tile_question_encoding')
        tiled_projected_input = tile_layer([encoded_answers, projected_input])
        similarity_layer = Lambda(lambda x: K.softmax(K.sum(x[0] * x[1], axis=2)),
                                  output_shape=lambda input_shapes: (input_shapes[0][0], input_shapes[0][1]),
                                  name='answer_similarity_softmax')
        softmax_output = similarity_layer([tiled_projected_input, encoded_answers])

        model = Model(input=[question_input_layer, answer_input_layer], output=softmax_output)
        return model

    def _instance_type(self):
        return QuestionAnswerInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'num_options': self.num_options,
                'answer_length': self.max_answer_length,
                }

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_answer_length = max_lengths['answer_length']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.max_sentence_length = self.model.get_input_shape_at(0)[1]
        # TODO(matt): implement this correctly
