from overrides import overrides

from keras.layers import Input

from .multiple_true_false_memory_network import MultipleTrueFalseMemoryNetwork
from ...training.models import DeepQaModel
from ...layers.attention import MaxSimilaritySoftmax
from ...layers.wrappers import EncoderWrapper


#TODO(pradeep): If we go down this route, properly merge this with memory networks.
class MultipleTrueFalseSimilarity(MultipleTrueFalseMemoryNetwork):
    '''
    This is a simple solver which chooses the option whose encoding is most similar to the
    background encoding. Whereas in a memory network solver there is a background summarization
    step based on the attention weights, this solver does not have an attention component, and
    merely compares the maximum of similarity scores with the background of all the options.
    Accordingly, all the parameters of the solver come from the encoder alone.

    While this class inherits MultipleTrueFalseMemoryNetwork for now, it is only the data
    preparation and encoding steps from that class that are reused here.
    '''

    @overrides
    def _build_model(self):
        question_input = Input(shape=self._get_question_shape(), dtype='int32', name="sentence_input")
        knowledge_input = Input(shape=self._get_background_shape(), dtype='int32', name="background_input")
        question_embedding = self._embed_input(question_input)
        knowledge_embedding = self._embed_input(knowledge_input)

        question_encoder = self._get_encoder()
        question_encoder = self._time_distribute_question_encoder(question_encoder)
        knowledge_encoder = EncoderWrapper(question_encoder, name='knowledge_encoder')
        encoded_question = question_encoder(question_embedding)  # (samples, num_options, encoding_dim)
        # (samples, num_options, knowledge_length, encoding_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)
        knowledge_axis = self._get_knowledge_axis()
        similarity_layer = MaxSimilaritySoftmax(knowledge_axis, self.max_knowledge_length, name="similarity_layer")
        option_probabilities = similarity_layer([encoded_question, encoded_knowledge])
        return DeepQaModel(input=[question_input, knowledge_input], output=option_probabilities)
