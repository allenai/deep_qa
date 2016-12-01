from typing import Any, Dict
from overrides import overrides

from keras.layers import Input, merge
from keras.engine import Layer

from .memory_network import MemoryNetworkSolver
from .multiple_true_false_memory_network import MultipleTrueFalseMemoryNetworkSolver
from ...layers.wrappers import TimeDistributedWithMask, EncoderWrapper
from ...layers.entailment_models import AnswerOptionSoftmax
from ...training.models import DeepQaModel


#TODO(pradeep): Properly merge this with memory networks.
class DecomposableAttentionSolver(MemoryNetworkSolver):
    '''
    This is a solver that embeds the question-option pair and the background and applies the
    decomposable attention entailment model (Parikh et al., EMNLP 2016). This currently only works
    with backgrounds of length=1 for now.

    While this class inherits MemoryNetworkSolver for now, it is only the data preparation and
    embedding steps from that class that are reused here.
    '''
    def __init__(self, params: Dict[str, Any]):
        super(DecomposableAttentionSolver, self).__init__(params)
        # This solver works only with decomposable_attention. Overwriting entailment_choices
        self.entailment_choices = ['decomposable_attention']

    @overrides
    def _build_model(self):
        question_input = Input(shape=self._get_question_shape(), dtype='int32', name="sentence_input")
        knowledge_input = Input(shape=self._get_background_shape(), dtype='int32', name="background_input")
        question_embedding = self._embed_input(question_input)
        knowledge_embedding = self._embed_input(knowledge_input)
        knowledge_embedding = TopKnowledgeSelector()(knowledge_embedding)

        entailment_layer = self._get_entailment_model()
        true_false_probabilities = entailment_layer([knowledge_embedding, question_embedding])
        return DeepQaModel(input=[question_input, knowledge_input], output=true_false_probabilities)


class MultipleTrueFalseDecomposableAttentionSolver(MultipleTrueFalseMemoryNetworkSolver):
    '''
    This solver extends the DecomposableAttentionSolver to the multiple true/false case.  Instead
    of just training and evaluating on true/false accuracy for each answer option separately, this
    model jointly makes one entailment decision for each answer option and is trained to rank the
    correct answer option as more entailed than the false answer options.

    We inherit from MultipleTrueFalseMemoryNetworkSolver to get a few convenience methods for
    dealing with multiple choice inputs (e.g., question shape, background shape, and loading data).
    '''
    def __init__(self, params: Dict[str, Any]):
        if 'entailment_model' not in params or 'final_activation' not in params['entailment_model']:
            params.setdefault('entailment_model', {})['final_activation'] = 'sigmoid'
        super(MultipleTrueFalseDecomposableAttentionSolver, self).__init__(params)
        # This solver works only with decomposable_attention. Overwriting entailment_choices
        self.entailment_choices = ['decomposable_attention']

    @overrides
    def _build_model(self):
        """
        Here we take a question and some background knowledge, restrict ourselves to looking at a
        single piece of the background knowledge (because the decomposable attention only works on
        a single sentence at the moment), then pass the question and the single background sentence
        through the decomposable attention entailment model as the hypothesis and the premise,
        respectively.

        We have to do a concat merge on the question and background sentence, in order to
        TimeDistribute the entailment model correctly.  Also note that we're TimeDistributing two
        of the layers in this model, but with _different_ masking behavior - when TimeDistributing
        the TopKnowledgeSelector, we want to TimeDistribute the mask computation also, as the
        output is still an embedded word sequence that needs its original mask.  When
        TimeDistributing the entailment layer, however, we want to mask whole answer options, like
        we do with encoders.  So we use different TimeDistributed subclasses to get different
        masking behavior for each of these cases.
        """
        # (batch_size, num_options, sentence_length)
        question_input = Input(shape=self._get_question_shape(), dtype='int32', name="sentence_input")
        # (batch_size, num_options, background_size, sentence_length)
        knowledge_input = Input(shape=self._get_background_shape(), dtype='int32', name="background_input")
        # (batch_size, num_options, sentence_length, embedding_size)
        question_embedding = self._embed_input(question_input)
        # (batch_size, num_options, background_size, sentence_length, embedding_size)
        knowledge_embedding = self._embed_input(knowledge_input)
        # (batch_size, num_options, sentence_length, embedding_size)
        knowledge_embedding = TimeDistributedWithMask(TopKnowledgeSelector())(knowledge_embedding)

        # (batch_size, num_options, sentence_length * 2, embedding_size)
        merged_embeddings = merge([knowledge_embedding, question_embedding], mode='concat',
                                  concat_axis=2)
        entailment_layer = EncoderWrapper(self._get_entailment_model())
        # (batch_size, num_options, 1)
        true_false_probabilities = entailment_layer(merged_embeddings)

        # (batch_size, num_options)
        final_softmax = AnswerOptionSoftmax()(true_false_probabilities)
        return DeepQaModel(input=[question_input, knowledge_input], output=final_softmax)


class TopKnowledgeSelector(Layer):
    '''
    Takes the embedding of (masked) knowledge, and returns the embedding of just the first sentence
    in knowledge.  We need this because DecomposableAttentionEntailment works with only one premise
    for now. We also assume here that the sentences in knowledge are sorted by their relevance.
    '''
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(TopKnowledgeSelector, self).__init__(**kwargs)

    def compute_mask(self, x, mask=None):
        # pylint: disable=unused-argument
        if mask is None:
            return None
        else:
            # input mask is of shape (batch_size, knowledge_length, sentence_length)
            return mask[:, 0, :]  #(batch_size, sentence_length)

    def get_output_mask_shape_for(self, input_shape):  # pylint: disable=no-self-use
        """
        This is a method I added in order to allow for proper mask computation in TimeDistributed.
        It is called by TimeDistributedWithMask.compute_mask() - see that code for some more
        insight as to why this is necessary.

        Once we're confident that this works, I plan on submitting a pull request to Keras with our
        improved TimeDistributed class.
        """
        # input_shape is (batch_size, knowledge_length, sentence_length, embed_dim)
        mask_shape = (input_shape[0], input_shape[2])
        return mask_shape

    def get_output_shape_for(self, input_shape):
        # input_shape is (batch_size, knowledge_length, sentence_length, embed_dim)
        return (input_shape[0], input_shape[2], input_shape[3])

    def call(self, x, mask=None):
        return x[:, 0, :, :]  # (batch_size, sentence_length, embed_dim)
