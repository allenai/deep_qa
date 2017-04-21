from typing import Dict
from overrides import overrides

from keras import backend as K
from keras.layers import Dense, Dropout, Input

from ...data.instances.multiple_choice_qa import QuestionAnswerInstance
from ...layers.wrappers import EncoderWrapper
from ...layers.attention import Attention
from ...training import TextTrainer
from ...common.params import Params
from ...training.models import DeepQaModel


class QuestionAnswerSimilarity(TextTrainer):
    """
    A TextTrainer that takes a question and several answer options as input, encodes the word
    sequences using a sentence encoder, optionally passes the question encoding through some dense
    layers, then selects the option that is most similar to the final question encoding.

    This assumes that you can get the parameters of the model to learn whatever associations
    between words in the question and words in the answer are necessary to select the correct
    choice.  There is no notion of external memory or background knowledge here.
    """
    def __init__(self, params: Params):
        self.num_hidden_layers = params.pop('num_hidden_layers', 1)
        self.hidden_layer_width = params.pop('hidden_layer_width', 50)
        self.hidden_layer_activation = params.pop('hidden_layer_activation', 'relu')
        self.max_answer_length = params.pop('max_answer_length', None)
        self.num_options = params.pop('num_options', None)
        super(QuestionAnswerSimilarity, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass both questions and answers through an embedding
        layer, then through an encoder.  Then we'll pass the encoded question through a some dense
        layers, and compare the similar to the encoded answers.
        """
        # First we create input layers and pass the inputs through an embedding.
        question_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="sentence_input")
        answer_input = Input(shape=(self.num_options,) + self._get_sentence_shape(self.max_answer_length),
                             dtype='int32',
                             name="answer_input")
        question_embedding = self._embed_input(question_input)
        answer_embedding = self._embed_input(answer_input)

        # Then we encode the question and answers using some encoder.
        question_encoder = self._get_encoder()
        encoded_question = question_encoder(question_embedding)
        # TODO(matt): make the dropout a parameter (should probably be "encoder_dropout", in
        # TextTrainer).
        regularized_encoded_question = Dropout(0.2)(encoded_question)

        # This needs to use a new encoder because we can't compile the same LSTM with two different
        # padding lengths...  If you really want to use the same LSTM for both questions and
        # answers, pad the answers to the same dimension as the questions, replacing
        # self.max_answer_length with self.num_sentence_words everywhere.
        answer_encoder = EncoderWrapper(question_encoder, name="answer_encoder")
        encoded_answers = answer_encoder(answer_embedding)

        # Then we pass the question through some hidden (dense) layers.
        hidden_input = regularized_encoded_question
        for i in range(self.num_hidden_layers):
            hidden_layer = Dense(units=self.hidden_layer_width,
                                 activation=self.hidden_layer_activation,
                                 name='question_hidden_layer_%d' % i)
            hidden_input = hidden_layer(hidden_input)
        projection_layer = Dense(units=K.int_shape(encoded_answers)[-1],
                                 activation='linear',
                                 name='question_projection')
        projected_input = projection_layer(hidden_input)

        # Lastly, we compare the similarity of the question to the answer options.  Note that this
        # layer has no parameters, so it doesn't need to be put into self._init_layers().
        softmax_output = Attention(name='answer_similarity_softmax')([projected_input, encoded_answers])

        return DeepQaModel(input=[question_input, answer_input], output=softmax_output)

    def _instance_type(self):
        return QuestionAnswerInstance

    @overrides
    def _get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super(QuestionAnswerSimilarity, self)._get_padding_lengths()
        padding_lengths['num_options'] = self.num_options
        padding_lengths['answer_length'] = self.max_answer_length
        return padding_lengths

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        super(QuestionAnswerSimilarity, self)._set_padding_lengths(padding_lengths)
        if self.max_answer_length is None:
            self.max_answer_length = padding_lengths['answer_length']
        if self.num_options is None:
            self.num_options = padding_lengths['num_options']

    @overrides
    def _set_padding_lengths_from_model(self):
        self.num_sentence_words = self.model.get_input_shape_at(0)[1]
        # TODO(matt): implement this correctly
