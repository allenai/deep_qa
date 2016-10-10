from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Lambda
from keras.models import Model

from .multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver


#TODO(pradeep): If we go down this route, properly merge this with memory networks.
class MultipleChoiceSimilaritySolver(MultipleChoiceMemoryNetworkSolver):
    '''
    This is a simple solver which chooses the option whose encoding is most similar to the background
    encoding. Whereas in a memory network solver there is a background summarization step based on the
    attention weights, this solver does not have an attention component, and merely compares the maximum
    of similarity scores with the background of all the options. Accordingly, all the parameters of the
    solver come from the encoder alone.

    While this class inherits MultipleChoiceMemoryNetworkSolver for now, it is only the data preparation
    and encoding steps from that class that are reused here.

    '''

    @overrides
    def _build_model(self):
        question_input_layer, question_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_question_shape(),
                name_prefix="sentence")
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(
                input_shape=self._get_background_shape(),
                name_prefix="background")
        question_encoder = self._get_sentence_encoder()
        knowledge_encoder = TimeDistributed(question_encoder, name='knowledge_encoder')
        encoded_question = question_encoder(question_embedding)  # (samples, num_options, encoding_dim)
        # (samples, num_options, knowledge_length, encoding_dim)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)
        knowledge_axis = self._get_knowledge_axis()
        similarity_function = _get_similarity_function(knowledge_axis, self.max_knowledge_length)
        similarity_layer = Lambda(similarity_function, output_shape=(self.num_options,), name="similarity_layer")
        option_probabilities = similarity_layer([encoded_question, encoded_knowledge])
        input_layers = [question_input_layer, knowledge_input_layer]
        similarity_solver = Model(input=input_layers, output=option_probabilities)
        return similarity_solver

# The following function needs to be outside the class definition to properly serialize Lambda layers.
# See Matt's comment here for more info: https://github.com/allenai/deep_qa/pull/90#issuecomment-252444488
def _get_similarity_function(knowledge_axis, max_knowledge_length):
    def func(question_knowledge_inputs):
        questions, knowledge = question_knowledge_inputs
        expanded_questions = K.expand_dims(questions, dim=knowledge_axis)
        # (samples, num_options, encoding_dim, knowledge_length)
        tiled_questions = K.tile(K.permute_dimensions(expanded_questions, (0, 1, 3, 2)),
                                 (1, 1, 1, max_knowledge_length))
        # (samples, num_options, knowledge_length, encoding_dim)
        tiled_questions = K.permute_dimensions(tiled_questions, (0, 1, 3, 2))
        # (samples, num_options, knowledge_length)
        question_knowledge_product = K.sum(tiled_questions * knowledge, axis=-1)
        max_knowledge_similarity = K.max(question_knowledge_product, axis=-1)  # (samples, num_options)
        return K.softmax(max_knowledge_similarity)
    return func
