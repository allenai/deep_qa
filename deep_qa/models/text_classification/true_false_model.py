from typing import Any, Dict
from overrides import overrides

from keras.layers import Dense, Dropout, Input

from ...data.instances.true_false_instance import TrueFalseInstance
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class TrueFalseModel(TextTrainer):
    """
    A TextTrainer that simply takes word sequences as input (could be either sentences or logical
    forms), encodes the sequence using a sentence encoder, then uses a few dense layers to decide
    if the sentence encoding is true or false.

    We don't really expect this model to work for question answering - it's just a sentence
    classification model.  The best it can do is basically to learn word cooccurrence information,
    similar to how the Salience solver works, and I'm not at all confident that this does that job
    better than Salience.  We've implemented this mostly as a simple baseline.

    Note that this also can't actually answer questions at this point.  You have to do some
    post-processing to get from true/false decisions to question answers, and I removed that from
    TextTrainer to make the code simpler.
    """
    def __init__(self, params: Dict[str, Any]):
        super(TrueFalseModel, self).__init__(params)

    @overrides
    def _build_model(self):
        '''
        train_input: numpy array: int32 (samples, num_words). Left padded arrays of word indices
            from sentences in training data
        '''
        # Step 1: Convert the sentence input into sequences of word vectors.
        sentence_input = Input(shape=self._get_sentence_shape(), dtype='int32', name="sentence_input")
        word_embeddings = self._embed_input(sentence_input)

        # Step 2: Pass the sequences of word vectors through the sentence encoder to get a sentence
        # vector..
        sentence_encoder = self._get_encoder()
        sentence_encoding = sentence_encoder(word_embeddings)

        # Add a dropout after LSTM.
        regularized_sentence_encoding = Dropout(0.2)(sentence_encoding)

        # Step 3: Find p(true | proposition) by passing the outputs from LSTM through an MLP with
        # ReLU layers.
        projection_layer = Dense(int(self.embedding_size/2), activation='relu', name='projector')
        softmax_layer = Dense(2, activation='softmax', name='softmax')
        output_probabilities = softmax_layer(projection_layer(regularized_sentence_encoding))

        # Step 4: Define crossentropy against labels as the loss.
        return DeepQaModel(input=sentence_input, output=output_probabilities)

    def _instance_type(self):
        return TrueFalseInstance

    @overrides
    def _set_max_lengths_from_model(self):
        self.set_text_lengths_from_model_input(self.model.get_input_shape_at(0)[1:])

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(TrueFalseModel, cls)._get_custom_objects()
        return custom_objects
