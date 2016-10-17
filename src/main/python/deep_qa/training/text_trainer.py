import logging
import pickle

from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy
from keras.layers import Dense, Input, Embedding, TimeDistributed, Dropout
from keras.models import Model
from overrides import overrides

from ..common.params import get_choice_with_default
from ..data.dataset import TextDataset
from ..data.instances.instance import Instance, TextInstance
from ..data.instances.true_false_instance import TrueFalseInstance
from ..data.embeddings import PretrainedEmbeddings
from ..data.tokenizer import tokenizers
from ..data.data_indexer import DataIndexer
from ..layers.encoders import encoders, set_regularization_params
from .trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextTrainer(Trainer):
    """
    This is a Trainer that deals with word sequences as its fundamental data type (any TextDataset
    or TextInstance subtype is fine).  That means we have to deal with padding, with converting
    words (or characters) to indices, and encoding word sequences.  This class adds methods on top
    of Trainer to deal with all of that stuff.
    """
    def __init__(self, params: Dict[str, Any]):

        # If specified, we will use the vectors in this file and learn a projection matrix to get
        # word vectors of dimension `embedding_size`, instead of learning the embedding matrix
        # ourselves.
        self.pretrained_embeddings_file = params.pop('pretrained_embeddings_file', None)

        # If we're using pre-trained embeddings, should we fine tune them?
        self.fine_tune_embeddings = params.pop('fine_tune_embeddings', False)

        # Should we have a projection layer on top of our embedding layer? (mostly useful with
        # pre-trained embeddings)
        self.project_embeddings = params.pop('project_embeddings', False)

        # Number of dimensions to use for word embeddings
        self.embedding_size = params.pop('embedding_size', 50)

        # Dropout parameter to apply to the word embedding layer
        self.embedding_dropout = params.pop('embedding_dropout', 0.5)

        # Upper limit on length of word sequences in the training data. Ignored during testing (we
        # use the value set at training time, either from this parameter or from a loaded model).
        # If this is not set, we'll calculate a max length from the data.
        self.max_sentence_length = params.pop('max_sentence_length', None)

        # Which tokenizer to use for TextInstances
        tokenizer_choice = get_choice_with_default(params, 'tokenizer', list(tokenizers.keys()))
        self.tokenizer = tokenizers[tokenizer_choice]()

        # These parameters specify the kind of encoder used to encode any word sequence input.
        # If given, this must be a dict.  We will use the "type" key in this dict (which must match
        # one of the keys in `encoders`) to determine the type of the encoder, then pass the
        # remaining args to the encoder constructor.
        # Hint: Use lstm or cnn for sentences, treelstm for logical forms, and bow for either.
        self.encoder_params = params.pop('encoder', {})

        super(TextTrainer, self).__init__(params)

        self.name = "TextTrainer"
        self.data_indexer = DataIndexer()

        # Model-specific member variables that will get set and used later.  For many of these, we
        # don't want to set them now, because they use max length information that only gets set
        # after reading the training data.
        self.embedding_layer = None
        self.projection_layer = None
        self.sentence_encoder_layer = None
        self._sentence_encoder_model = None

    @overrides
    def _prepare_data(self, dataset: TextDataset, for_train: bool):
        """
        Takes dataset, which could be a complex tuple for some classes, and produces as output a
        tuple of (inputs, labels), which can be used directly with Keras to either train or
        evaluate self.model.
        """
        if for_train:
            self.data_indexer.fit_word_dictionary(dataset)
        logger.info("Indexing dataset")
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        max_lengths = self._get_max_lengths()
        logger.info("Padding dataset to lengths %s", str(max_lengths))
        indexed_dataset.pad_instances(max_lengths)
        if for_train:
            self._set_max_lengths(indexed_dataset.max_lengths())
        inputs, labels = indexed_dataset.as_training_data()
        if isinstance(inputs[0], tuple):
            inputs = [numpy.asarray(x) for x in zip(*inputs)]
        else:
            inputs = numpy.asarray(inputs)
        return inputs, numpy.asarray(labels)

    @overrides
    def _prepare_instance(self, instance: TextInstance, make_batch: bool=True):
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad(self._get_max_lengths())
        inputs, label = indexed_instance.as_training_data()
        if make_batch:
            if isinstance(inputs, tuple):
                inputs = [numpy.expand_dims(x, axis=0) for x in inputs]
            else:
                inputs = numpy.expand_dims(inputs, axis=0)
        return inputs, label

    @overrides
    def _process_pretraining_data(self):
        """
        Adds words to the vocabulary based on the data used by the pretrainers.  We want this to
        happen before loading the training data so that we can use pretraining to expand our
        applicable vocabulary.
        """
        logger.info("Fitting the data indexer using the pretraining data")
        for pretrainer in self.pretrainers:
            pretrainer.fit_data_indexer()

    def _load_layers(self):
        """
        We have some variables that store individual layers used by the model, so that they can be
        re-used in several places if desired.  When we load a model, we have to set those layers,
        or things might break in really odd ways.  This method is in charge of finding those
        layers and initializing their variables.

        Note that this specifically looks for the layers defined by _get_embedded_sentence_input
        and _get_sentence_encoder.  If you change any of that in a subclass, or add other layers
        that are re-used, you must override this method, or loading models will break.  Similarly,
        if you change code in those two methods (e.g., making the sentence encoder into two
        layers), this method must be changed accordingly.

        Note that we don't need to store any TimeDistributed() layers directly, because they don't
        have any parameters themselves.
        """
        logger.info("Loading individual layers from model for re-use")

        # I guess we could just load the embedding layer, the projection layer, and the sentence
        # encoder, and reconstruct the others, because they don't actually have any parameters...
        # But, we'll stick with this for now.
        for layer in self.model.layers:
            if layer.name == "embedding":
                logger.info("  Found embedding layer")
                self.embedding_layer = layer
            elif layer.name == "sentence_embedding":
                logger.info("  Found wrapped embedding layer")
                if self.embedding_layer is None:
                    self.embedding_layer = layer.layer
                else:
                    logger.warning("  FOUND DUPLICATE EMBEDDING LAYER!  NOT SURE WHAT TO DO!")
            elif layer.name == "embedding_projection":
                logger.info("  Found projection layer")
                self.projection_layer = layer
            elif layer.name == "sentence_encoder":
                logger.info("  Found sentence encoder")
                self.sentence_encoder_layer = layer
            elif layer.name == "timedist_sentence_encoder":
                logger.info("  Found sentence encoder")
                if self.sentence_encoder_layer is None:
                    self.sentence_encoder_layer = layer
                else:
                    logger.warning("  FOUND DUPLICATE SENTENCE ENCODER LAYER!  NOT SURE WHAT TO DO!")
        assert self.embedding_layer is not None, "Embedding layer not found"
        assert self.sentence_encoder_layer is not None, "Sentence encoder not found"
        if self.project_embeddings:
            assert self.projection_layer is not None, "Projection layer not found"

    def get_sentence_vector(self, sentence: str):
        """
        Given a sentence (just a string), use the model's sentence encoder to convert it into a
        vector.  This is mostly just useful for debugging.
        """
        if self._sentence_encoder_model is None:
            self._build_sentence_encoder_model()
        instance = TrueFalseInstance(sentence, True, tokenizer=self.tokenizer)
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad({'word_sequence_length': self.max_sentence_length})
        instance_input, _ = indexed_instance.as_training_data()
        encoded_instance = self._sentence_encoder_model.predict(numpy.asarray([instance_input]))
        return encoded_instance[0]

    def _get_max_lengths(self) -> Dict[str, int]:
        """
        This is about padding.  Any solver will have some number of things that need padding in
        order to make a compilable model, like the length of a sentence.  This method returns a
        dictionary of all of those things, mapping a length key to an int.
        """
        raise NotImplementedError

    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        """
        This is about padding.  Any solver will have some number of things that need padding in
        order to make a compilable model, like the length of a sentence.  This method sets those
        variables given a dictionary of lengths, perhaps computed from training data or loaded from
        a saved model.
        """
        raise NotImplementedError

    @overrides
    def _set_params_from_model(self):
        self._set_max_lengths_from_model()

    @overrides
    def _save_auxiliary_files(self):
        super(TextTrainer, self)._save_auxiliary_files()
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        data_indexer_file.close()

    @overrides
    def _load_auxiliary_files(self):
        super(TextTrainer, self)._load_auxiliary_files()
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        data_indexer_file.close()

    def _set_max_lengths_from_model(self):
        """
        Given a loaded model, set the max_lengths needed for padding.  This is necessary so that we
        can pad the test data if we just loaded a saved model.
        """
        raise NotImplementedError

    def _instance_type(self) -> Instance:
        """
        When reading datasets, what instance type should we create?
        """
        raise NotImplementedError

    def _load_dataset_from_files(self, files: List[str]):
        """
        This method assumes you have a TextDataset that can be read from a single file.  If you
        have something more complicated, you'll need to override this method (though, a solver that
        has background information could call this method, then do additional processing on the
        rest of the list, for instance).
        """
        return TextDataset.read_from_file(files[0], self._instance_type(), tokenizer=self.tokenizer)

    def _get_embedded_sentence_input(self, input_shape: Tuple[int], name_prefix: str):
        """
        Performs the initial steps of embedding sentences.  This function takes care of two steps:
        (1) it creates an Input layer for the sentence input, and (2) it converts word indices into
        word vectors with an Embedding layer.  Depending on how this object was initialized, we
        might initialize that Embedding layer from pre-trained word vectors, or add a projection on
        top of the embedding, or add dropout after the embedding layer.

        We use `input_shape` to decide how many TimeDistributed() layers we need on top of the
        initial embedding / projection.  We will always do one less TimeDistributed() layer than
        there are dimensions in `input_shape`.  For example, if you pass an input shape of
        (max_sentence_length,), we will not add any TimeDistributed() layers.  If you pass
        (max_knowledge_length, max_sentence_length), we'll add one, and if you pass (num_options,
        max_knowledge_length, max_sentence_length), we'll add two, etc.

        Because Keras requires access to the actual Input() layer, we return that along with the
        final word vector layer.
        """
        # pylint: disable=redefined-variable-type
        input_layer = Input(shape=input_shape, dtype='int32', name=name_prefix + "_input")

        # If a particular model has several places where sentences are encoded, we want to be sure
        # to use the _same_ embedding layer for all of them, so we initialize them and save them to
        # self here, if we haven't already done so.
        if self.embedding_layer is None:
            if self.pretrained_embeddings_file:
                # If we have a pre-trained embeddings file, we'll create the embedding layer
                # initialized with the embeddings in the file.  These embeddings can either be
                # fixed or tunable.
                self.embedding_layer = PretrainedEmbeddings.get_embedding_layer(
                        self.pretrained_embeddings_file,
                        self.data_indexer,
                        self.fine_tune_embeddings)
            else:
                self.embedding_layer = Embedding(input_dim=self.data_indexer.get_vocab_size(),
                                                 output_dim=self.embedding_size,
                                                 mask_zero=True,  # this handles padding correctly
                                                 name='embedding')
            if self.project_embeddings:
                self.projection_layer = TimeDistributed(Dense(output_dim=self.embedding_size,),
                                                        name='embedding_projection')

        # Now we actually embed the input and apply dropout.
        embedding = self.embedding_layer
        for _ in input_shape[1:]:
            embedding = TimeDistributed(embedding, name=name_prefix + "_embedding")
        embedded_input = embedding(input_layer)

        if self.project_embeddings:
            projection = self.projection_layer
            for _ in input_shape[1:]:
                projection = TimeDistributed(projection, name=name_prefix + "_projection")
            embedded_input = projection(embedded_input)

        if self.embedding_dropout > 0.0:
            embedded_input = Dropout(self.embedding_dropout, name=name_prefix + "_dropout")(embedded_input)

        return input_layer, embedded_input

    def _get_sentence_encoder(self):
        """
        A sentence encoder takes as input a sequence of word embeddings, and returns as output a
        single vector encoding the sentence.  This is typically either a simple RNN or an LSTM, but
        could be more complex, if the "sentence" is actually a logical form.
        """
        if self.sentence_encoder_layer is None:
            self.sentence_encoder_layer = self._get_new_encoder()
        return self.sentence_encoder_layer

    def _get_new_encoder(self, name="sentence_encoder"):
        # The code that follows would be destructive to self.encoder_params (lots of calls to
        # params.pop()), but we may need to create several encoders.  So we'll make a copy and use
        # that instead of self.encoder_params.
        encoder_params = deepcopy(self.encoder_params)
        encoder_type = get_choice_with_default(encoder_params, "type", list(encoders.keys()))
        encoder_params["name"] = name
        encoder_params["output_dim"] = self.embedding_size
        set_regularization_params(encoder_type, encoder_params)
        return encoders[encoder_type](**encoder_params)

    def _build_sentence_encoder_model(self):
        """
        Here we pull out just a couple of layers from self.model and use them to define a
        stand-alone encoder model.

        Specifically, we need the part of the model that gets us from word index sequences to word
        embedding sequences, and the part of the model that gets us from word embedding sequences
        to sentence vectors.

        This must be called after self.max_sentence_length has been set, which happens when
        self._get_training_data() is called.
        """
        input_layer, embedded_input = self._get_embedded_sentence_input(
                input_shape=(self.max_sentence_length,), name_prefix="sentence")
        encoder_layer = self._get_sentence_encoder()
        encoded_input = encoder_layer(embedded_input)
        self._sentence_encoder_model = Model(input=input_layer, output=encoded_input)

        # Loss and optimizer do not matter here since we're not going to train this model. But it
        # needs to be compiled to use it for prediction.
        self._sentence_encoder_model.compile(loss="mse", optimizer="adam")
        self._sentence_encoder_model.summary()

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(TextTrainer, cls)._get_custom_objects()
        for value in encoders.values():
            if value.__name__ not in ['LSTM']:
                custom_objects[value.__name__] = value
        return custom_objects
