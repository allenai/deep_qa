from typing import Any, Dict
from overrides import overrides

from keras.layers import Input, merge

from .multiple_true_false_memory_network import MultipleTrueFalseMemoryNetwork
from ...layers.attention.masked_softmax import MaskedSoftmax
from ...layers.top_knowledge_selector import TopKnowledgeSelector
from ...layers.wrappers.time_distributed_with_mask import TimeDistributedWithMask
from ...layers.wrappers.encoder_wrapper import EncoderWrapper
from ...training.models import DeepQaModel

class MultipleTrueFalseDecomposableAttention(MultipleTrueFalseMemoryNetwork):
    '''
    This solver extends the DecomposableAttention to the multiple true/false case.  Instead
    of just training and evaluating on true/false accuracy for each answer option separately, this
    model jointly makes one entailment decision for each answer option and is trained to rank the
    correct answer option as more entailed than the false answer options.

    We inherit from MultipleTrueFalseMemoryNetwork to get a few convenience methods for
    dealing with multiple choice inputs (e.g., question shape, background shape, and loading data).
    '''
    def __init__(self, params: Dict[str, Any]):
        if 'entailment_model' not in params or 'final_activation' not in params['entailment_model']:
            params.setdefault('entailment_model', {})['final_activation'] = 'sigmoid'
        super(MultipleTrueFalseDecomposableAttention, self).__init__(params)
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
        # (batch_size, num_options, sentence_length, embedding_dim)
        question_embedding = self._embed_input(question_input)
        # (batch_size, num_options, background_size, sentence_length, embedding_dim)
        knowledge_embedding = self._embed_input(knowledge_input)
        # (batch_size, num_options, sentence_length, embedding_dim)
        knowledge_embedding = TimeDistributedWithMask(TopKnowledgeSelector())(knowledge_embedding)

        # (batch_size, num_options, sentence_length * 2, embedding_dim)
        merged_embeddings = merge([knowledge_embedding, question_embedding], mode='concat',
                                  concat_axis=2)
        entailment_layer = EncoderWrapper(self._get_entailment_model())
        # (batch_size, num_options, 1)
        true_false_probabilities = entailment_layer(merged_embeddings)

        # (batch_size, num_options)
        final_softmax = MaskedSoftmax()(true_false_probabilities)
        return DeepQaModel(input=[question_input, knowledge_input], output=final_softmax)
