'''
A basic knowledge encoder takes a arbitrary encoder (LSTM, BOW) and applies it to each of the background
sentences independently. These classes are designed to extend the functionality of the basic
question encoder to allowpost-processing of the knowledge before it is used in the Memory
Network hops.
'''

from collections import OrderedDict
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import GRU

from .additive import Additive
from .wrappers.encoder_wrapper import EncoderWrapper


class IndependentKnowledgeEncoder:
    '''
    An Independent KnowledgeEncoder simply wraps EncoderWrapper around the question encoder.
    Given a question encoder, which takes (samples, sentence_length, embedding_dim), we
    want to be able to use the question encoder on the background sentences which accompany the
    question, of shape (samples, knowledge_length, sentence_length, embedding_dim).
    Therefore, to apply the question encoder to all of the background knowledge, we simply
    TimeDistribute it. The 'time' dimension is assumed to be 1.
    '''
    # pylint: disable=unused-argument
    def __init__(self, question_encoder, name, **kwargs):
        self.question_encoder = question_encoder
        self.name = name
        self.encoder_wrapper = EncoderWrapper(self.question_encoder, name=self.name)

    def __call__(self, knowledge_embedding):
        return self.encoder_wrapper(knowledge_embedding)


class TemporalKnowledgeEncoder(IndependentKnowledgeEncoder):
    """
    This class implements the "temporal encoding" from the end-to-end memory networks paper, which
    adds a learned vector to the encoding of each time step in the memory.
    """
    def __init__(self, **kwargs):
        super(TemporalKnowledgeEncoder, self).__init__(**kwargs)
        self.additive_layer = Additive(name=self.name + "_additive")

    def __call__(self, knowledge_embedding):
        encoded_knowledge = self.encoder_wrapper(knowledge_embedding)
        return self.additive_layer(encoded_knowledge)


class BiGRUKnowledgeEncoder(IndependentKnowledgeEncoder):
    '''
    A BiGRUKnowledgeEncoder performs the same inital encoding as the IndependentKnowledgeEncoder,
    but applies a BiDirectional GRU over the encoded knowledge in order to allow the order of the
    background knowledge to be relevant to the downstream solver. This implementation follows the
    Fusion layer in the Dynamic Memory Networks paper: https://arxiv.org/pdf/1603.01417v1.pdf.

    First, we apply the wrapped question encoder:
    (samples, knowledge_length, sentence_length, embedding_dim) => (samples, knowlege_length, encoding_dim).

    Then, we run the encoded background sentence vectors through  a BiDirectional GRU, where the
    time dimension is the knowledge_length (this allows the Memory Network to take into account
    that the background knowledge may have some temporal structure, such as in the Babi tasks).

    (samples, knowledge_length, encoding_dim) => (samples, knowledge_length, encoding_dim)

    Note that in Keras, we must explicitly set the 'return_sequences' flag in order to return the
    full set of vectors, rather than just the last one in the sequence. Additionally, the merge
    mode we choose for this BiDirectional GRU is 'sum', rather than the standard 'concat' which
    would double the encoding_dim.

    Additionally, if we are using a MultipleChoice Memory Network, we have multiple sets of
    background knowledge related to each answer. If this is the case, then we again TimeDistribute
    the BiDirectional GRU to apply this to every set of background knowledge.
    '''
    def __init__(self, **kwargs):
        self.knowledge_length = kwargs.pop("knowledge_length")
        self.encoding_dim = kwargs.pop("encoding_dim")
        self.has_multiple_backgrounds = kwargs.pop("has_multiple_backgrounds")
        super(BiGRUKnowledgeEncoder, self).__init__(**kwargs)
        # TODO: allow the merge_mode of the GRU/other parameters to be passed as arguments.
        self.bi_gru = Bidirectional(GRU(self.encoding_dim, return_sequences=True),
                                    input_shape=(self.knowledge_length, self.encoding_dim),
                                    merge_mode='sum', name='{}_bi_gru'.format(self.name))
        if self.has_multiple_backgrounds:
            # pylint: disable=redefined-variable-type
            self.bi_gru = EncoderWrapper(self.bi_gru, name='wrapped_{}'.format(self.name))

    def __call__(self, knowledge_embedding):
        base_time_distributed_layer = super(BiGRUKnowledgeEncoder, self).__call__(knowledge_embedding)
        return self.bi_gru(base_time_distributed_layer)


knowledge_encoders = OrderedDict()  # pylint:  disable=invalid-name
knowledge_encoders['independent'] = IndependentKnowledgeEncoder
knowledge_encoders['temporal'] = TemporalKnowledgeEncoder
knowledge_encoders['bi_gru'] = BiGRUKnowledgeEncoder
