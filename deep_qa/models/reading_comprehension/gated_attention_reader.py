from typing import Any, Dict
from overrides import overrides
from keras.layers import Input, Dropout, merge
from keras.callbacks import LearningRateScheduler

from ...data.instances.mc_question_answer_instance import McQuestionAnswerInstance
from ...common.checks import ConfigurationError
from ...layers.backend.batch_dot import BatchDot
from ...layers.attention.attention import Attention
from ...layers.attention.masked_softmax import MaskedSoftmax
from ...layers.option_attention_sum import OptionAttentionSum
from ...layers.overlap import Overlap
from ...layers.attention.gated_attention import GatedAttention
from ...layers.l1_normalize import L1Normalize
from ...layers.vector_matrix_split import VectorMatrixSplit
from ...layers.bigru_index_selector import BiGRUIndexSelector
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class GatedAttentionReader(TextTrainer):
    """
    This TextTrainer implements the Gated Attention Reader model described in
    "Gated-Attention Readers for Text Comprehension" by Dhingra et. al 2016. It encodes
    the document with a variable number of gated attention layers, and then encodes
    the query. It takes the dot product of these two final encodings to generate an
    attention over the words in the document, and it then selects the option with the
    highest summed or mean weight as the answer.

    Parameters
    ----------
    multiword_option_mode: str, optional (default="mean")
        Describes how to calculate the probability of options
        that contain multiple words. If "mean", the probability of
        the option is taken to be the mean of the probabilities of
        its constituent words. If "sum", the probability of the option
        is taken to be the sum of the probabilities of its constituent
        words.

    num_gated_attention_layers: int, optional (default=3)
        The number of gated attention layers to pass the document
        embedding through. Must be at least 1.

    cloze_token: str, optional (default=None)
        If not None, the string that represents the cloze token in a cloze question.
        Used to calculate the attention over the document, as the model does it
        differently for cloze vs non-cloze datasets.

    gating_function: str, optional (default="*")
        The gating function to use in the Gated Attention layer. ``"*"`` is for
        elementwise multiplication, ``"+"`` is for elementwise addition, and
        ``"|"`` is for concatenation.

    gated_attention_dropout: float, optional (default=0.3)
        The proportion of units to drop out after each gated attention layer.

    qd_common_feature: boolean, optional (default=True)
        Whether to use the question-document common word feature. This feature simply
        indicates, for each word in the document, whether it appears in the query
        and has been shown to improve reading comprehension performance.
    """
    def __init__(self, params: Dict[str, Any]):
        self.max_question_length = params.pop('max_question_length', None)
        self.max_passage_length = params.pop('max_passage_length', None)
        self.max_option_length = params.pop('max_option_length', None)
        self.num_options = params.pop('num_options', None)
        # either "mean" or "sum"
        self.multiword_option_mode = params.pop('multiword_option_mode', "mean")
        # number of gated attention layers to use
        self.num_gated_attention_layers = params.pop('num_gated_attention_layers', 3)
        # gating function to use, either "*", "+", or "|"
        self.gating_function = params.pop('gating_function', "*")
        # dropout proportion after each gated attention layer.
        self.gated_attention_dropout = params.pop('gated_attention_dropout', 0.3)
        # If you are using the model on a cloze (fill in the blank) dataset,
        # indicate what token indicates the blank.
        self.cloze_token = params.pop('cloze_token', None)
        self.cloze_token_index = None
        # use the question document common word feature
        self.use_qd_common_feature = params.pop('qd_common_feature', True)
        super(GatedAttentionReader, self).__init__(params)

    @overrides
    def _build_model(self):
        """
        The basic outline here is that we'll pass the questions and the
        document / passage (think of this as a collection of possible answer
        choices) into a word embedding layer.
        """
        # get the index of the cloze token, if applicable
        if self.cloze_token is not None:
            self.cloze_token_index = self.data_indexer.get_word_index(self.cloze_token)

        # First we create input layers and pass the question and document
        # through embedding layers.

        # shape: (batch size, question_length)
        question_input_shape = self._get_sentence_shape(self.max_question_length)
        question_input = Input(shape=question_input_shape,
                               dtype='int32', name="question_input")

        # if using character embeddings, split off the question word indices.
        if len(question_input_shape) > 1:
            question_indices = VectorMatrixSplit(split_axis=-1)(question_input)[0]
        else:
            question_indices = question_input

        # shape: (batch size, document_length)
        document_input_shape = self._get_sentence_shape(self.max_passage_length)
        document_input = Input(shape=self._get_sentence_shape(self.max_passage_length),
                               dtype='int32',
                               name="document_input")

        # if using character embeddings, split off the document word indices.
        if len(document_input_shape) > 1:
            document_indices = VectorMatrixSplit(split_axis=-1)(document_input)[0]
        else:
            document_indices = document_input
        # shape: (batch size, number of options, num words in option)
        options_input_shape = ((self.num_options,) +
                               self._get_sentence_shape(self.max_option_length))
        options_input = Input(shape=options_input_shape,
                              dtype='int32', name="options_input")

        # if using character embeddings, split off the option word indices.
        if len(options_input_shape) > 2:
            options_indices = VectorMatrixSplit(split_axis=-1)(options_input)[0]
        else:
            options_indices = options_input

        # shape: (batch size, question_length, embedding size)
        question_embedding = self._embed_input(question_input, embedding_name="question_embedding")

        # shape: (batch size, document_length, embedding size)
        document_embedding = self._embed_input(document_input, embedding_name="document_embedding")

        # We pass the question and document embedding through a variable
        # number of gated-attention layers.
        if self.num_gated_attention_layers < 1:
            raise ConfigurationError("Need at least one gated attention layer.")
        for i in range(self.num_gated_attention_layers-1):
            # Note that the size of the last dimension of the input
            # is not necessarily the embedding size in the second gated
            # attention layer and beyond.

            # We encode the question embeddings with a seq2seq encoder.
            question_encoder = self._get_seq2seq_encoder(name="question_{}".format(i))
            # shape: (batch size, question_length, 2*seq2seq hidden size)
            encoded_question = question_encoder(question_embedding)

            # We encode the document embeddings with a seq2seq encoder.
            # Note that this is not the same encoder as used for the question.
            document_encoder = self._get_seq2seq_encoder(name="document_{}".format(i))
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            encoded_document = document_encoder(document_embedding)

            # (batch size, document length, question length)
            qd_attention = BatchDot()([encoded_document, encoded_question])
            # (batch size, document length, question length)
            normalized_qd_attention = MaskedSoftmax()([qd_attention])

            gated_attention_layer = GatedAttention(self.gating_function,
                                                   name="gated_attention_{}".format(i))
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            document_embedding = gated_attention_layer([encoded_document,
                                                        encoded_question,
                                                        normalized_qd_attention])
            gated_attention_dropout = Dropout(self.gated_attention_dropout)
            # shape: (batch size, document_length, 2*seq2seq hidden size)
            document_embedding = gated_attention_dropout([document_embedding])

        # Last Layer
        if self.use_qd_common_feature:
            # get the one-hot features for common occurence
            # shape (batch size, document_indices, 2)
            qd_common_feature = Overlap()([document_indices,
                                           question_indices])
            # We concatenate qd_common_feature with the document embeddings.
            # shape: (batch size, document_length, (2*seq2seq hidden size) + 2)
            document_embedding = merge([document_embedding, qd_common_feature],
                                       mode='concat')
        # We encode the document embeddings with a final seq2seq encoder.
        document_encoder = self._get_seq2seq_encoder(name="document_final")
        # shape: (batch size, document_length, 2*seq2seq hidden size)
        final_encoded_document = document_encoder(document_embedding)

        if self.cloze_token is None:
            # Get a final encoding of the question from a biGRU that does not return
            # the sequence, and use it to calculate attention over the document.
            final_question_encoder = self._get_encoder(name="question_final")
            # shape: (batch size, 2*seq2seq hidden size)
            final_encoded_question = final_question_encoder(question_embedding)
        else:
            # We get a final encoding of the question by concatenating the forward
            # and backward GRU at the index of the cloze token.
            final_question_encoder = self._get_seq2seq_encoder(name="question_final")
            # each are shape (batch size, question_length, seq2seq hidden size)
            encoded_question_f, encoded_question_b = final_question_encoder(question_embedding)
            # extract the gru outputs at the cloze token from the forward and
            # backwards passes
            index_selector = BiGRUIndexSelector(self.cloze_token_index)
            final_encoded_question = index_selector([question_indices,
                                                     encoded_question_f,
                                                     encoded_question_b])

        # take the softmax of the document_embedding after it has been passed
        # through gated attention layers to get document probabilities
        # shape: (batch size, document_length)
        document_probabilities = Attention(name='question_document_softmax')([final_encoded_question,
                                                                              final_encoded_document])
        # We sum together the weights of words that match each option
        # and use the multiword_option_mode to determine how to calculate
        # the total probability of the option.
        options_sum_layer = OptionAttentionSum(self.multiword_option_mode,
                                               name="options_probability_sum")
        # shape: (batch size, num_options)
        options_probabilities = options_sum_layer([document_indices,
                                                   document_probabilities,
                                                   options_indices])

        # We normalize the option_probabilities by dividing it by its L1 norm.
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
        max_lengths = super(GatedAttentionReader, self)._get_max_lengths()
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
        super(GatedAttentionReader, self)._set_max_lengths(max_lengths)
        self.max_question_length = max_lengths['num_question_words']
        self.max_passage_length = max_lengths['num_passage_words']
        self.max_option_length = max_lengths['num_option_words']
        self.num_options = max_lengths['num_options']

    @overrides
    def _set_max_lengths_from_model(self):
        self.num_sentence_words = self.model.get_input_shape_at(0)[1]
        # TODO(matt): implement this correctly

    @overrides
    def _get_callbacks(self):
        callbacks = super(GatedAttentionReader, self)._get_callbacks()
        def _half_lr(epoch):
            initial_lr = 0.0005
            if epoch > 3:
                return initial_lr / ((epoch-3)*.5)
            return initial_lr
        lrate = LearningRateScheduler(_half_lr)
        callbacks.append(lrate)
        return callbacks

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(GatedAttentionReader, cls)._get_custom_objects()
        custom_objects["Attention"] = Attention
        custom_objects["BatchDot"] = BatchDot
        custom_objects["BiGRUIndexSelector"] = BiGRUIndexSelector
        custom_objects["GatedAttention"] = GatedAttention
        custom_objects["L1Normalize"] = L1Normalize
        custom_objects["MaskedSoftmax"] = MaskedSoftmax
        custom_objects["OptionAttentionSum"] = OptionAttentionSum
        custom_objects["Overlap"] = Overlap
        custom_objects["VectorMatrixSplit"] = VectorMatrixSplit

        return custom_objects
