from keras.layers import Dense, Input, TimeDistributed, Multiply, Average

from overrides import overrides

from deep_qa.common.params import Params
from deep_qa.data.instances.sequence_tagging import concrete_instances
from deep_qa.training.text_trainer import TextTrainer
from deep_qa.training.models import DeepQaModel
from deep_qa.layers.attention import WeightedSum
from deep_qa.layers.encoders.bag_of_words import BOWEncoder


class VerbSemanticsModel(TextTrainer):
    """
    This ``VerbSemanticsModel`` takes as input a sentence and query.
    Query includes verb and entity spans in the sentence.
    The model predicts (state-change-type, argument-tags)
    where, state-change-type: denotes what is happening to the entity, e.g. CREATE/DESTROY/MOVE
    argument-tags: denotes arguments of the state change, e.g. source and destination location of the entity.

    We use BOW encoder along with a dense layer to predict state-change-type.
    We use stacked seq2seq encoders followed by a dense layer to predict argument-tags.

    Parameters
    ----------
    num_stacked_rnns : int, optional (default: ``1``)
        The number of ``seq2seq_encoders`` that we should stack on top of each other before
        predicting tags.
    instance_type : str
        Specifies the instance type, currently the only supported type is "VerbSemanticsInstance",
        which defines things like how the input data is formatted and tokenized.
    """
    def __init__(self, params: Params):
        self.num_stacked_rnns = params.pop('num_stacked_rnns', 1)
        instance_type = params.pop('instance_type', "VerbSemanticsInstance")
        self.instance_type = concrete_instances[instance_type]
        super(VerbSemanticsModel, self).__init__(params)

    @overrides
    def _instance_type(self):
        return self.instance_type

    @overrides
    def _build_model(self):
        # Input: (sentence, verb-span, entity-span)
        # Output: (state-change-type, argument-tags)

        # shape: (batch_size, text_length)
        sentence_input = Input(shape=self._get_sentence_shape(), dtype='int32', name='word_array_input')
        verb_input = Input(shape=self._get_sentence_shape(), dtype='int32', name='verb_array_input')
        entity_input = Input(shape=self._get_sentence_shape(), dtype='int32', name='entity_array_input')

        # shape: (batch_size, text_length, embedding_dim)
        sentence_embedding = self._embed_input(sentence_input)
        multiply_layer = Multiply()
        verb_indices = multiply_layer([sentence_input, verb_input])
        verb_embedding = self._embed_input(verb_indices)
        entity_indices = multiply_layer([sentence_input, entity_input])
        entity_embedding = self._embed_input(entity_indices)

        # For state-change-type prediction: We first convert verb and entity into bag-of-words(BOW) representation
        # and then apply a dense layer with soft-max activation to predict state change type.
        bow_features = BOWEncoder()
        average_layer = Average()
        verb_entity_vector = average_layer([bow_features(verb_embedding), bow_features(entity_embedding)])
        state_change_type = Dense(self.data_indexer.get_vocab_size("state_changes") - 2,
                                  activation='softmax')(verb_entity_vector)

        # For argument-tags prediction: We first convert a sentence to a sequence of word embeddings
        # and then apply a stack of seq2seq encoders to finally predict a sequence of argument tags.
        for i in range(self.num_stacked_rnns):
            encoder = self._get_seq2seq_encoder(name="encoder_{}".format(i),
                                                fallback_behavior="use default params")
            # shape still (batch_size, text_length, embedding_dim)
            print("Inside _build_model:", sentence_embedding)
            sentence_embedding = encoder(sentence_embedding)

        # The -2 below is because we are ignoring the padding and unknown tokens that the
        # DataIndexer has by default.
        argument_tags = TimeDistributed(Dense(self.data_indexer.get_vocab_size('tags') - 2,
                                              activation='softmax'))(sentence_embedding)

        return DeepQaModel(input=[sentence_input, verb_input, entity_input],
                           output=[state_change_type, argument_tags])

    @overrides
    def _set_padding_lengths_from_model(self):
        # We return the dimensions of
        # 0th layer which is "indexed input" (0),
        # 0th item in the input tuple which is "sentence" [0],
        # and everything that comes after the batch_size which "includes #words, #characters etc." [1:]
        self._set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[0][1:])

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(VerbSemanticsModel, cls)._get_custom_objects()
        # If we use any custom layers implemented in deep_qa (not part of original Keras),
        # they need to be added in the custom_objects dictionary.
        custom_objects["WeightedSum"] = WeightedSum
        custom_objects["BOWEncoder"] = BOWEncoder
        return custom_objects
