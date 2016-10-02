import logging

import pickle
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import numpy
from keras.layers import Dense, Input, Embedding, TimeDistributed, Dropout
from keras.models import Model, model_from_json

from ..common.params import get_choice_with_default
from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from ..data.text_instance import TrueFalseInstance, TextInstance
from ..data.embeddings import PretrainedEmbeddings
from ..data.tokenizer import tokenizers
from ..data.data_indexer import DataIndexer
from ..layers.encoders import encoders, set_regularization_params

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NNSolver:
    def __init__(self, params: Dict[str, Any]):
        # Prefix for saving and loading model files
        self.model_prefix = params.pop('model_serialization_prefix')
        parent_directory = os.path.dirname(self.model_prefix)
        os.makedirs(parent_directory, exist_ok=True)

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

        # Upper limit on the number of training instances.  If this is set, and we get more than
        # this, we will truncate the data.
        self.max_training_instances = params.pop('max_training_instances', None)

        self.train_file = params.pop('train_file', None)
        self.positive_train_file = params.pop('positive_train_file', None)
        self.negative_train_file = params.pop('negative_train_file', None)
        self.validation_file = params.pop('validation_file', None)
        self.test_file = params.pop('test_file', None)

        # Visualize the intermediate outputs of the trained model. Output will be written to
        # <model_serialization_prefix>_debug_<epoch>.txt
        self.debug_file = params.pop('debug_file', None)

        # Which tokenizer to use for TextInstances
        tokenizer_choice = get_choice_with_default(params, 'tokenizer', list(tokenizers.keys()))
        self.tokenizer = tokenizers[tokenizer_choice]()

        # Amount of training data to use for Keras' validation (not our QA validation, set by
        # the validation_file param, which is separate).  This value is passed as
        # 'validation_split' to Keras' model.fit().
        self.keras_validation_split = params.pop('keras_validation_split', 0.1)
        # Number of train epochs.
        self.num_epochs = params.pop('num_epochs', 20)
        # Number of epochs to be patient before early stopping.
        self.patience = params.pop('patience', 1)

        # These parameters specify the kind of encoder used to encode any word sequence input.
        # If given, this must be a dict.  We will use the "type" key in this dict (which must match
        # one of the keys in `encoders`) to determine the type of the encoder, then pass the
        # remaining args to the encoder constructor.
        # Hint: Use lstm or cnn for sentences, treelstm for logical forms, and bow for either.
        self.encoder_params = params.pop('encoder', {})

        # We've now processed all of the parameters, and we're the base class, so there should not
        # be anything left.
        assert len(params.keys()) == 0, "You passed unrecognized parameters: " + str(params)

        self.data_indexer = DataIndexer()

        # Model-specific member variables that will get set and used later.  For many of these, we
        # don't want to set them now, because they use max length information that only gets set
        # after reading the training data.
        self.model = None
        self.debug_model = None
        self.embedding_layer = None
        self.projection_layer = None
        self.sentence_encoder_layer = None
        self._sentence_encoder_model = None

        # Training-specific member variables that will get set and used later.
        self.best_epoch = -1
        self.pretrainers = []
        # We store the datasets used for training and validation, both before processing and after
        # processing, in case a subclass wants to modify it between epochs for whatever reason.
        self.training_dataset = None
        self.train_input = None
        self.train_labels = None
        self.validation_dataset = None
        self.validation_input = None
        self.validation_labels = None

    def _instance_type(self):
        """
        When reading datasets, what instance type should we create?
        """
        raise NotImplementedError

    def prep_labeled_data(self, dataset: TextDataset, for_train: bool, shuffle: bool):
        """
        Takes dataset, which could be a complex tuple for some classes, and produces as output a
        tuple of (inputs, labels), which can be used directly with Keras to either train or
        evaluate self.model.

        For training and validation data, this method is called internally during self.train().  If
        you want to evaluate the model on some other test dataset, this is the method you need to
        call.  However, that dataset has to have labels, or this method will crash.  We don't
        currently have an API for making predictions on data that doesn't have labels.  TODO(matt)
        """
        processed_dataset = self._index_and_pad_dataset(dataset, self._get_max_lengths())
        if for_train:
            self._set_max_lengths(processed_dataset.max_lengths())
        inputs, labels = processed_dataset.as_training_data(shuffle)
        if isinstance(inputs[0], tuple):
            inputs = [numpy.asarray(x) for x in zip(*inputs)]
        else:
            inputs = numpy.asarray(inputs)
        return inputs, numpy.asarray(labels)

    def can_train(self) -> bool:
        """
        Returns True if we were given enough inputs to train the model, False otherwise.
        """
        has_train_file = (self.train_file is not None) or (self.positive_train_file is not None and
                                                           self.negative_train_file is not None)
        has_validation_file = self.validation_file is not None
        return has_train_file and has_validation_file

    def can_test(self) -> bool:
        """
        Return True if we were given enough inputs to test the model, False otherwise.
        """
        return self.test_file is not None

    def _load_pretraining_data(self):
        """
        Adds words to the vocabulary based on the data used by the pretrainers.  We want this to
        happen before loading the training data so that we can use pretraining to expand our
        applicable vocabulary.
        """
        for pretrainer in self.pretrainers:
            pretrainer.fit_data_indexer(self.data_indexer)

    def _pretrain(self):
        """
        Runs whatever pre-training has been specified in the constructor.
        """
        for pretrainer in self.pretrainers:
            pretrainer.train()

    def train(self):
        '''
        Trains the model.

        All training parameters have already been passed to the constructor, so we need no
        arguments to this method.
        '''
        logger.info("Running training")

        # Before actually doing any training, we'll run whatever pre-training has been specified.
        # Note that this can have funny interactions with the data indexer, which typically gets
        # fit to the training data.  We'll take the apporach of having the pre-trainer also fit the
        # data indexer on whatever data it uses, as pre-trainers typically train encoder models,
        # which encludes word embeddings.  Fitting the data indexer again when loading the actual
        # training data won't hurt anything.
        self._load_pretraining_data()

        # First we need to prepare the data that we'll use for training.
        logger.info("Getting training data")
        self.train_input, self.train_labels = self._get_training_data()
        logger.info("Getting validation data")
        self.validation_input, self.validation_labels = self._get_validation_data()

        # We need to actually do pretraining _after_ we've loaded the training data, though, as we
        # need to build the models to be consistent between training and pretraining.  The training
        # data tells us a max sentence length, which we need for the pretrainer.
        self._pretrain()

        # Then we build the model and compile it.
        logger.info("Building the model")
        self.model = self._build_model()
        self.model.summary()
        # TODO(pradeep): Try out other optimizers, especially rmsprop.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self.debug_file:
            # Get the list of layers whose outputs will be visualized as per the
            # solver definition and build a debug model.
            debug_layers = self.get_debug_layer_names()
            self.debug_model = self._build_debug_model(debug_layers)
            self.debug_model.compile(loss='mse', optimizer='sgd')  # Will not train this model.
            debug_dataset, debug_input = self._get_debug_dataset_and_input()

        # Now we actually train the model, with patient early stopping using the validation data.
        best_accuracy = 0.0
        self.best_epoch = 0
        num_worse_epochs = 0
        for epoch_id in range(self.num_epochs):
            self._pre_epoch_hook(epoch_id)
            logger.info("Epoch %d", epoch_id)
            kwargs = {'nb_epoch': 1}
            if self.keras_validation_split > 0.0:
                kwargs['validation_split'] = self.keras_validation_split
            self.model.fit(self.train_input, self.train_labels, **kwargs)
            logger.info("Running validation")
            accuracy = self.evaluate(self.validation_labels, self.validation_input)
            logger.info("Validation accuracy: %.4f", accuracy)
            if accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= self.patience:
                    logger.info("Stopping training")
                    break
            else:
                best_accuracy = accuracy
                self.best_epoch = epoch_id
                num_worse_epochs = 0  # Reset the counter.
                self._save_model(epoch_id)
            if self.debug_file:
                # Shows intermediate outputs of the model on validation data
                self.debug(debug_dataset, debug_input, epoch_id)
        self._save_best_model()

    def test(self):
        """
        Tests the model, using the file given to the constructor.
        """
        inputs, labels = self._get_test_data()
        logger.info("Scoring test data")
        accuracy = self.evaluate(labels, inputs)
        logger.info("Test accuracy: %.4f", accuracy)

    def _get_debug_dataset_and_input(self):
        # TODO(matt): Return validation dataset by default
        raise NotImplementedError

    def debug(self, debug_dataset, debug_input, epoch: int):
        """
        Each solver defines its own debug method to visualize the
        appropriate layers.
        """
        raise NotImplementedError

    def get_debug_layer_names(self):
        """
        Each solver defines its own list of layers whose output will
        be visualized.
        """
        raise NotImplementedError

    def load_model(self, epoch: int=None):
        """
        Loads a serialized model.  If epoch is not None, we try to load the model from that epoch.
        If epoch is not given, we load the best saved model.

        Paths in here must match those in self._save_model(epoch) and self._save_best_model(), or
        things will break.
        """
        logger.info("Loading serialized model")
        # Loading serialized model
        model_config_file = open("%s_config.json" % self.model_prefix)
        model_config_json = model_config_file.read()
        self.model = model_from_json(model_config_json,
                                     custom_objects=self._get_custom_objects())
        if epoch is not None:
            model_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, epoch)
        else:
            model_file = "%s_weights.h5" % self.model_prefix
        logger.info("Loading weights from file %s", model_file)
        self.model.load_weights(model_file)
        self.model.summary()
        self._load_layers()
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        model_config_file.close()
        data_indexer_file.close()
        self._set_max_lengths_from_model()

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

    def score_instance(self, instance: TextInstance):
        indexed_instance = instance.to_indexed_instance(self.data_indexer)
        indexed_instance.pad(self._get_max_lengths())
        inputs, _ = indexed_instance.as_training_data()
        if isinstance(inputs, tuple):
            inputs = [numpy.expand_dims(x, axis=0) for x in inputs]
        else:
            inputs = numpy.expand_dims(inputs, axis=0)
        return self.score(inputs)

    def score(self, test_input):
        return self.model.predict(test_input)

    @staticmethod
    def group_by_question(labels):
        """
        This method takes a sequential numpy array of shape (num_instances, 2), and groups it by
        question, resulting in an array of shape (num_questions).  This method works when the input
        consists of labels produced by self.prep_labeled_data() or when it is predictions produced
        by self.model.predict().

        To get to the desired output, we do a few steps.  First, we convert the input labels to
        (num_instances, 1), by dropping the first index in the second dimension.  We do this
        because the second dimension is [prob false, prob true], and we only care about prob true
        here.  Then we reshape the input to (num_questions, 4) and find the index of the
        highest-scoring answer, returning an array of shape (num_questions).  This allows us to
        compute question accuracy, instead of an instance-level loss function.

        We assume that the data that produced `labels` has
        dataset.can_be_converted_to_multiple_choice() == True.  See the comments there for the
        requirements on the input data.

        TODO(matt): remove this code and just use MultipleChoiceInstance instead.  This will
        require an option to flatten the training inputs (or maybe we just do it here?).
        """
        num_questions = int(len(labels) / 4)
        reshaped_labels = labels[:, 1].reshape(num_questions, 4)
        return numpy.argmax(reshaped_labels, axis=1)

    def evaluate(self, labels, test_input):
        """
        Given ground-truth labels for which answer option is correct, compute question accuracy.

        labels: a numpy array of shape (num_questions), where the value is an index indicating the
            correct answer index for each question.
        test_input: input values that will be scored with self.model.predict().  Must have length
            num_questions * 4.

        We will score the test input using the model, group the scores by question, then compute
        accuracy.
        """
        test_scores = self.score(test_input)
        test_predictions = self.group_by_question(test_scores)
        num_correct = sum(test_predictions == labels)
        accuracy = float(num_correct) / len(test_predictions)
        return accuracy

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

    def _set_max_lengths_from_model(self):
        """
        Given a loaded model, set the max_lengths needed for padding.  This is necessary so that we
        can pad the test data if we just loaded a saved model.
        """
        raise NotImplementedError

    def _get_training_data(self):
        """Loads training data and converts it into a format suitable for input to Keras.  This
        method must return a tuple of (train_input, train_labels).  This method also must set
        self.training_dataset with the unprocessed Dataset object.

        This method takes no arguments; any necessary arguments (e.g., a path for where to find the
        training data) must have been passed to the constructor of this object.

        This base implementation is suitable for NNSolvers that only take single sentences /
        logical forms as input.  NNSolvers that have more complicated inputs will need to override
        this method.
        """
        instance_type = self._instance_type()
        if self.train_file:
            dataset = TextDataset.read_from_file(self.train_file, instance_type, tokenizer=self.tokenizer)
        else:
            positive_dataset = TextDataset.read_from_file(self.positive_train_file,
                                                          instance_type,
                                                          label=True,
                                                          tokenizer=self.tokenizer)
            negative_dataset = TextDataset.read_from_file(self.negative_train_file,
                                                          instance_type,
                                                          label=False,
                                                          tokenizer=self.tokenizer)
            dataset = positive_dataset.merge(negative_dataset)
        if self.max_training_instances is not None:
            logger.info("Truncating the dataset to %d instances", self.max_training_instances)
            dataset = dataset.truncate(self.max_training_instances)
        self.data_indexer.fit_word_dictionary(dataset)
        self.training_dataset = dataset
        return self.prep_labeled_data(dataset, for_train=True, shuffle=True)

    def _get_validation_data(self):
        """
        Like _get_training_data, but for validation data.  Also, while _get_training_data() returns
        instance-level train_labels, for computing gradients during training, for validation we
        group the labels by question, to use question accuracy as our early stopping criterion.

        This base implementation is suitable for NNSolvers that only take single sentences /
        logical forms as input.  NNSolvers that have more complicated inputs will need to override
        this method.
        """
        self.validation_dataset = TextDataset.read_from_file(self.validation_file,
                                                             self._instance_type(),
                                                             tokenizer=self.tokenizer)
        return self._prep_question_dataset(self.validation_dataset)

    def _get_test_data(self):
        dataset = TextDataset.read_from_file(self.test_file, self._instance_type(), tokenizer=self.tokenizer)
        return self._prep_question_dataset(dataset)

    def _prep_question_dataset(self, dataset: TextDataset):
        assert dataset.can_be_converted_to_multiple_choice(), "Dataset not formatted as questions"
        inputs, labels = self.prep_labeled_data(dataset, for_train=False, shuffle=False)
        return inputs, self.group_by_question(labels)

    def _index_and_pad_dataset(self, dataset: TextDataset, max_lengths: Dict[str, int]):
        logger.info("Indexing dataset")
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        logger.info("Padding dataset to lengths %s", str(max_lengths))
        indexed_dataset.pad_instances(max_lengths)
        return indexed_dataset

    def _build_model(self) -> Model:
        """Constructs and returns a Keras model that will take the output of
        self._get_training_data as input, and produce as output a true/false decision for each
        input.

        The returned model will be used to call model.fit(train_input, train_labels).
        """
        raise NotImplementedError

    def _build_debug_model(self, debug_layer_names: List[str]):
        """
        Accesses self.model and extracts the necessary parts of the model out to define another
        model that has the intermediate outputs we want to visualize.
        """
        debug_inputs = self.model.get_input_at(0)  # list of all input_layers
        debug_outputs = []
        for layer in self.model.layers:
            if layer.name in debug_layer_names:
                debug_outputs.append(layer.get_output_at(0))
        debug_model = Model(input=debug_inputs, output=debug_outputs)
        return debug_model

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

    def _pre_epoch_hook(self, epoch: int):
        """
        This method gets called before each epoch of training.  If a solver wants to do any kind of
        processing in between epochs (e.g., updating the training data for whatever reason), here
        is your chance to do so.
        """
        pass

    def _save_model(self, epoch: int):
        # Serializing the model for future use.
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json" % (self.model_prefix), "w")
        print(model_config, file=model_config_file)
        self.model.save_weights("%s_weights_epoch=%d.h5" % (self.model_prefix, epoch), overwrite=True)
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_file)
        model_config_file.close()
        data_indexer_file.close()

    def _save_best_model(self):
        '''Copies the weights from the best epoch to a final weight file

        The point of this is so that the input/output spec of the NNSolver is simpler.  Someone
        calling this as a subroutine doesn't have to worry about which epoch ended up being the
        best, they can just use the final weight file.  You can still use models from other epochs
        if you really want to.
        '''
        from shutil import copyfile
        epoch_weight_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, self.best_epoch)
        final_weight_file = "%s_weights.h5" % self.model_prefix
        copyfile(epoch_weight_file, final_weight_file)

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = {}
        for value in encoders.values():
            if value.__name__ not in ['LSTM']:
                custom_objects[value.__name__] = value
        return custom_objects
