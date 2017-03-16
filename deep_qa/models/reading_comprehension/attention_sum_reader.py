from typing import Any, Dict
from overrides import overrides
from keras.layers import Input

from ...data.instances.mc_question_answer_instance import McQuestionAnswerInstance
from ...layers.attention.attention import Attention
from ...layers.option_attention_sum import OptionAttentionSum
from ...layers.l1_normalize import L1Normalize
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class AttentionSumReader(TextTrainer):
    """
    This TextTrainer implements the Attention Sum Reader model described by
    Kadlec et. al 2016. It takes a question and document as input, encodes the
    document and question words with two separate Bidirectional GRUs, and then
    takes the dot product of the question embedding with the document embedding
    of each word in the document. This creates an attention over words in the
    document, and it then selects the option with the highest summed or mean
    weight as the answer.
    """
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_passage_length = params.pop('max_passage_length', None)
        self.max_option_length = params.pop('max_option_length', None)
        self.num_options = params.pop('num_options', None)
        # either "mean" or "sum"
        self.multiword_option_mode = params.pop('multiword_option_mode', "mean")
        super(AttentionSumReader, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass the questions and the
        document / passage (think of this as a collection of possible answer
        choices) into a word embedding layer.

        Then, we run the word embeddings from the document (a sequence) through
        a bidirectional GRU and output a sequence that is the same length as
        the input sequence size. For each time step, the output item
        ("contextual embedding") is the concatenation of the forward and
        backward hidden states in the bidirectional GRU encoder at that time
        step.

        To get the encoded question, we pass the words of the question into
        another bidirectional GRU. This time, the output encoding is a vector
        containing the concatenation of the last hidden state in the forward
        network with the last hidden state of the backward network.

        We then take the dot product of the question embedding with each of the
        contextual embeddings for the words in the documents. We sum up all the
        occurences of a word ("total attention"), and pick the word with the
        highest total attention in the document as the answer.
        """
        # First we create input layers and pass the inputs through embedding layers.

        # shape: (batch size, question_length)
        question_input = Input(shape=self._get_sentence_shape(self.max_question_length),
                               dtype='int32', name="question_input")
        # shape: (batch size, document_length)
        document_input = Input(shape=self._get_sentence_shape(self.max_passage_length),
                               dtype='int32',
                               name="document_input")
        # shape: (batch size, num_options, options_length)
        options_input = Input(shape=(self.num_options,) + self._get_sentence_shape(self.max_option_length),
                              dtype='int32', name="options_input")
        # shape: (batch size, question_length, embedding size)
        question_embedding = self._embed_input(question_input)

        # shape: (batch size, document_length, embedding size)
        document_embedding = self._embed_input(document_input)

        # We encode the question embeddings with some encoder.
        question_encoder = self._get_encoder()
        # shape: (batch size, 2*embedding size)
        encoded_question = question_encoder(question_embedding)

        # We encode the document with a seq2seq encoder. Note that this is not the same encoder as
        # used for the question.
        # TODO(nelson): Enable using the same encoder for both document and question.  (This would
        # be hard in our current code; you would need a method to transform an encoder into a
        # seq2seq encoder.)
        document_encoder = self._get_seq2seq_encoder()
        # shape: (batch size, document_length, 2*embedding size)
        encoded_document = document_encoder(document_embedding)

        # Here we take the dot product of `encoded_question` and each word
        # vector in `encoded_document`.
        # shape: (batch size, max document length in words)
        document_probabilities = Attention(name='question_document_softmax')([encoded_question,
                                                                              encoded_document])
        # We sum together the weights of words that match each option.
        options_sum_layer = OptionAttentionSum(self.multiword_option_mode,
                                               name="options_probability_sum")
        # shape: (batch size, num_options)
        options_probabilities = options_sum_layer([document_input,
                                                   document_probabilities, options_input])
        # We normalize the option_probabilities by dividing each
        # element by L1 norm (sum) of the whole tensor.
        l1_norm_layer = L1Normalize()

        # shape: (batch size, num_options)
        option_normalized_probabilities = l1_norm_layer(options_probabilities)
        return DeepQaModel(input=[question_input, document_input, options_input],
                           output=option_normalized_probabilities)

    @overrides
    def _instance_type(self):
        """
        Return the instance type that the model trains on.
        """
        return McQuestionAnswerInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        """
        Return a dictionary with the appropriate padding lengths.
        """
        max_lengths = super(AttentionSumReader, self)._get_max_lengths()
        max_lengths['num_question_words'] = self.max_question_length
        max_lengths['num_passage_words'] = self.max_passage_length
        max_lengths['num_option_words'] = self.max_option_length
        max_lengths['num_options'] = self.num_options
        return max_lengths

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        """
        Set the padding lengths of the model.
        """
        # TODO(nelson): superclass complains that there is no
        # num_sentence_words key, so we set it to None here.
        # We should probably patch up / organize the API.
        max_lengths["num_sentence_words"] = None
        super(AttentionSumReader, self)._set_max_lengths(max_lengths)
        self.max_question_length = max_lengths['num_question_words']
        self.max_passage_length = max_lengths['num_passage_words']
        self.max_option_length = max_lengths['num_option_words']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[1][1:])
        self.max_question_length = self.model.get_input_shape_at(0)[0][1]
        self.max_passage_length = self.model.get_input_shape_at(0)[1][1]
        self.num_options = self.model.get_input_shape_at(0)[2][1]
        self.max_option_length = self.model.get_input_shape_at(0)[2][2]

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(AttentionSumReader, cls)._get_custom_objects()
        custom_objects["Attention"] = Attention
        custom_objects["L1Normalize"] = L1Normalize
        custom_objects["OptionAttentionSum"] = OptionAttentionSum
        return custom_objects
