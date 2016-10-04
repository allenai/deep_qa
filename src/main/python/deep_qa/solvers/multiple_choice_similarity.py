from typing import Dict
from overrides import overrides

from keras import backend as K
from keras.layers import TimeDistributed, Lambda, merge
from keras.models import Model

from .multiple_choice_memory_network import MultipleChoiceMemoryNetworkSolver


#TODO(pradeep): If we go down this route, properly merge this with memory networks.
class MultipleChoiceSimilaritySolver(MultipleChoiceMemoryNetworkSolver):
    '''
    This is a simple solver which chooses the otption whose encoding is most similar to the background
    encoding. Whereas in a memory network solver there is a background summarization step based on the
    attention weights, this solver does not have an attention component, and merely compares the maximum
    of similarity scores with the background of all the options. Accordingly, all the parameters of the 
    solver come from the encoder alone.

    While this class inherits MultipleChoiceMemoryNetworkSolver for now, it is only the data preparation
    and encoding steps from that class that are reused here.

    '''
    
    @overrides
    def _build_model(self):
        question_input_layer, question_embedding = self._get_embedded_sentence_input(input_shape=self._get_question_shape(),
                                                                                     name_prefix="sentence")
        knowledge_input_layer, knowledge_embedding = self._get_embedded_sentence_input(input_shape=self._get_background_shape(),
                                                                                     name_prefix="background")
        question_encoder = self._get_sentence_encoder()
        knowledge_encoder = TimeDistributed(question_encoder, name='knowledge_encoder')
        encoded_question = question_encoder(question_embedding)  # (samples, num_options, word_dim) 
        encoded_knowledge = knowledge_encoder(knowledge_embedding)  # (samples, num_options, knowledge_length, word_dim)
        knowledge_axis = self._get_knowledge_axis()
        merge_mode = lambda layer_outs: K.concatenate([K.expand_dims(layer_outs[0], dim=knowledge_axis),
                                                       layer_outs[1]], axis=knowledge_axis)
        merged_shape = self._get_merged_background_shape()
        # (samples, num_options, knowledge_length + 1, word_dim)
        merged_question_knowledge = merge([encoded_question, encoded_knowledge], mode=merge_mode,
                                          output_shape=merged_shape, name='concat_question_with_background')
        def _similarity_function(merged_question_knowledge):
            expanded_questions = merged_question_knowledge[:, :, :1, :]  # (samples, num_options, 1, word_dim)
            # (samples, num_options, word_dim, knowledge_length)
            tiled_questions = K.tile(K.permute_dimensions(expanded_questions, (0, 1, 3, 2)), (self.max_knowledge_length,))
            tiled_questions = K.permute_dimensions(tiled_questions, (0, 1, 3, 2))  # (samples, num_options, knowledge_length, word_dim)
            knowledge = merged_question_knowledge[:, :, 1:, :]  # (samples, num_options, knowledge_length, word_dim)
            question_knowledge_product = K.sum(tiled_questions * knowledge, axis=-1)  # (samples, num_options, knowledge_length)
            max_knowledge_similarity = K.max(question_knowledge_product, axis=-1)  # (samples, num_options)
            return K.softmax(max_knowledge_similarity)
        option_knowledge_similarity_layer = Lambda(_similarity_function, output_shape=(self.num_options,), name="similarity_layer")
        option_probabilities = option_knowledge_similarity_layer(merged_question_knowledge)
        input_layers = [question_input_layer, knowledge_input_layer]
        similarity_solver = Model(input=input_layers, output=option_probabilities)
        return similarity_solver
        
