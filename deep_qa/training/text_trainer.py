from copy import deepcopy
from typing import Any, Dict, List, Tuple
import logging

import dill as pickle
from keras import backend as K
from keras.layers import Dense, Dropout, Layer, TimeDistributed
from overrides import overrides
import numpy
import tensorflow

from ..common.checks import ConfigurationError
from ..common.params import Params
from ..common.util import clean_layer_name
from ..data import tokenizers, DataIndexer, DataGenerator, IndexedDataset, TextDataset
from ..data.embeddings import PretrainedEmbeddings
from ..data.instances import Instance, TextInstance
from ..data.datasets import concrete_datasets
from ..layers import TimeDistributedEmbedding
from ..layers.encoders import encoders, set_regularization_params, seq2seq_encoders
from .trainer import Trainer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class TextTrainer(Trainer):
    # pylint: disable=line-too-long
    """
    This is a Trainer that deals with word sequences as its fundamental data type (any TextDataset
    or TextInstance subtype is fine).  That means we have to deal with padding, with converting
    words (or characters) to indices, and encoding word sequences.  This class adds methods on top
    of Trainer to deal with all of that stuff.

    This class has five kinds of methods:

    (1) protected methods that are overriden from :class:`~deep_qa.training.trainer.Trainer`, and
        which you shouldn't need to worry about
    (2) utility methods for building models, intended for use by subclasses
    (3) abstract methods that determine a few key points of behavior in concrete subclasses (e.g.,
        what your input data type is)
    (4) model-specific methods that you `might` have to override, depending on what your model
        looks like - similar to (3), but simple models don't need to override these
    (5) private methods that you shouldn't need to worry about

    There are two main ways you're intended to interact with this class, then: by calling the
    utility methods when building your model, and by customizing the behavior of your concrete
    model by using the parameters to this class.

    Parameters
    ----------
    embeddings : Dict[str, Any], optional (default=50 dim word embeddings, 8 dim character
    embeddings, 0.5 dropout on both)
        These parameters specify the kind of embeddings to use for words, character, tags, or
        whatever you want to embed.  This dictionary behaves similarly to the ``encoder`` and
        ``seq2seq_encoder`` parameter dictionaries.  Valid keys are ``dimension``, ``dropout``,
        ``pretrained_file``, ``fine_tune``, and ``project``.  The value for ``dimension`` is an
        ``int`` specifying the dimensionality of the embedding (default 50 for words, 8 for
        characters); ``dropout`` is a float, specifying the amount of dropout to use on the
        embedding layer (default ``0.5``); ``pretrained_file`` is a (string) path to a glove-formatted file
        containing pre-trained embeddings; ``fine_tune`` is a boolean specifying whether the
        pretrained embeddings should be trainable (default ``False``); and ``project`` is a boolean
        specifying whether to add a projection layer after the embedding layer (only really useful
        in conjunction with pre-trained embeddings, to get them into a lower-dimensional space;
        default ``False``).
    data_generator: Dict[str, Any], optional (default=None)
        If not ``None``, we will pass these parameters to a :class:`DataGenerator` object to create
        data batches, instead of creating one big array for all of our training data.  See
        :class:`DataGenerator` for the available options here.  Note that in order to take full
        advantage of the capabilities of a ``DataGenerator``, you should make sure your model
        correctly implements :func:`~TextTrainer._set_padding_lengths`,
        :func:`~TextTrainer.get_padding_lengths`,
        :func:`~TextTrainer.get_padding_memory_scaling`, and
        :func:`~TextTrainer.get_instance_sorting_keys`.  Also note that some of the things
        ``DataGenerator`` does can change the behavior of your learning algorithm, so you should
        think carefully about how exactly you want batches to be structured before you choose these
        parameters.
    num_sentence_words: int, optional (default=None)
        Upper limit on length of word sequences in the training data. Ignored during testing (we
        use the value set at training time, either from this parameter or from a loaded model).  If
        this is not set, we'll calculate a max length from the data.
    num_word_characters: int, optional (default=None)
        Upper limit on length of words in the training data. Only applicable for "words and
        characters" text encoding.
    tokenizer: Dict[str, Any], optional (default={})
        Which tokenizer to use for ``TextInstances``.  See
        :mod:``deep_qa.data.tokenizers.tokenizer`` for more information.
    encoder: Dict[str, Dict[str, Any]], optional (default={'default': {}})
        These parameters specify the kind of encoder used to encode any word sequence input.  An
        encoder takes a sequence of vectors and returns a single vector.

        If given, this must be a dict, where each key is a name that can be used for encoders in
        the model, and the value corresponding to the key is a set of parameters that will be
        passed on to the constructor of the encoder.  We will use the "type" key in this dict
        (which must match one of the keys in `encoders`) to determine the type of the encoder, then
        pass the remaining args to the encoder constructor.

        Hint: Use ``"lstm"`` or ``"cnn"`` for sentences, ``"treelstm"`` for logical forms, and
        ``"bow"`` for either.
    encoder_fallback_behavior: string, optional (default="crash")
        Determines the behavior when an encoder is asked for by name, but you have not given
        parameters for an encoder with that name.  See ``_get_encoder`` for more information.
    seq2seq_encoder: Dict[str, Dict[str, Any]], optional (default={'default': {'encoder_params': {}, 'wrapper_params: {}}})
        Like ``encoder``, except seq2seq encoders return a sequence of vectors instead of a single
        vector (the difference between our "encoders" and "seq2seq encoders" is the difference in
        Keras between ``LSTM()`` and ``LSTM(return_sequences=True)``).
    seq2seq_encoder_fallback_behavior: string, optional (default="crash")
        Determines the behavior when a seq2seq encoder is asked for by name, but you have not given
        parameters for an encoder with that name.  See ``_get_seq2seq_encoder`` for more
        information.
    """
    # pylint: enable=line-too-long
    def __init__(self, params: Params):
        self.embedding_params = params.pop('embeddings',
                                           {'words': {'dim': 50, 'dropout': 0.5},
                                            'characters': {'dim': 8, 'dropout': 0.5}})
        self.pretrained_param = None
        self.use_rand = None
        self.test_embeddings = params.pop('test_embeddings', False)
        self.embeddings = None

        data_generator_params = params.pop('data_generator', None)
        if data_generator_params is not None:
            self.data_generator = DataGenerator(self, data_generator_params)
        else:
            self.data_generator = None

        self.dataset_params = params.pop("dataset", {})
        dataset_type_key = self.dataset_params.pop_choice("type", list(concrete_datasets.keys()),
                                                          default_to_first_choice=True)
        self.dataset_type = concrete_datasets[dataset_type_key]
        self.num_sentence_words = params.pop('num_sentence_words', None)
        self.num_word_characters = params.pop('num_word_characters', None)

        tokenizer_params = params.pop('tokenizer', {})
        tokenizer_choice = tokenizer_params.pop_choice('type', list(tokenizers.keys()),
                                                       default_to_first_choice=True)
        self.tokenizer = tokenizers[tokenizer_choice](tokenizer_params)
        # Note that the way this works is a little odd - we need each Instance object to do the
        # right thing when we call instance.words() and instance.to_indexed_instance().  So we set
        # a class variable on TextInstance so that _all_ TextInstance objects use the setting that
        # we read here.
        TextInstance.tokenizer = self.tokenizer

        self.encoder_params = params.pop('encoder', {'default': {}})
        fallback_choices = ['crash', 'use default encoder', 'use default params']
        self.encoder_fallback_behavior = params.pop_choice('encoder_fallback_behavior', fallback_choices,
                                                           default_to_first_choice=True)
        self.seq2seq_encoder_params = params.pop('seq2seq_encoder',
                                                 {'default': {"encoder_params": {},
                                                              "wrapper_params": {}}})
        self.seq2seq_encoder_fallback_behavior = params.pop_choice('seq2seq_encoder_fallback_behavior',
                                                                   fallback_choices,
                                                                   default_to_first_choice=True)

        super(TextTrainer, self).__init__(params)

        self.name = "TextTrainer"
        self.data_indexer = DataIndexer()

        # These keep track of which names you've used to get embeddings and encoders, so that we
        # reuse layers that you want to reuse.
        self.embedding_layers = {}
        self.encoder_layers = {}
        self.seq2seq_encoder_layers = {}

    ###########################
    # Overriden Trainer methods - you shouldn't have to worry about these, though for some
    # advanced uses you might override some of them, especially _get_custom_objects.
    ###########################

    @overrides
    def create_data_arrays(self, dataset: IndexedDataset):
        if self.data_generator is not None:
            return self.data_generator.create_generator(dataset)
        else:
            dataset.pad_instances(self.get_padding_lengths())
            return dataset.as_training_data()

    @overrides
    def load_dataset_from_files(self, files: List[str]):
        """
        This method assumes you have a TextDataset that can be read from a single file.  If you
        have something more complicated, you'll need to override this method (though, a solver that
        has background information could call this method, then do additional processing on the
        rest of the list, for instance).
        """
        dataset_params = deepcopy(self.dataset_params)
        return self.dataset_type.read_from_file(files[0], self._instance_type(), dataset_params)

    @overrides
    def score_dataset(self, dataset: TextDataset):
        """
        See the superclass docs (:func:`Trainer.score_dataset`) for usage info.  Just a note here
        that we `do not` use data generators for this method, even if you've said elsewhere that
        you want to use them, so that we can easily return the labels for the data.  This means
        that we'll do whole-dataset padding, and this could be slow.  We could probably fix this,
        but it's good enough for now.
        """
        # TODO(matt): for some reason the reference to the super class docs above isn't getting
        # linked properly.  I'm guessing it's because of an indexing issue in sphinx, but I
        # couldn't figure it out.  Once that works, it can be changed to "See :func:`the superclass
        # docs <Trainer.score_dataset>` for usage info").
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        # Because we're not using data generators here, we need to save and hide
        # `self.data_generator`.  TODO(matt): it _should_ be as easy as iterating over the data
        # again to pull out the labels, so we can still use data generators, but I'm waiting on
        # implementing that.
        data_generator = self.data_generator
        self.data_generator = None
        inputs, labels = self.create_data_arrays(indexed_dataset)
        predictions = self.model.predict(inputs)
        self.data_generator = data_generator
        return predictions, labels

    @overrides
    def set_model_state_from_dataset(self, dataset: TextDataset):
        logger.info("Fitting data indexer word dictionary.")
        self.data_indexer.fit_word_dictionary(dataset)

    @overrides
    def set_model_state_from_file(self, vocab_file: TextDataset):
        logger.info("Fitting data indexer with vocabulary file.")
        self.data_indexer.set_from_file(vocab_file)

    @overrides
    def set_model_state_from_indexed_dataset(self, dataset: IndexedDataset):
        self._set_padding_lengths(dataset.padding_lengths())

    def _dataset_indexing_kwargs(self) -> Dict[str, Any]:
        return {'data_indexer': self.data_indexer}

    @overrides
    def _set_params_from_model(self):
        self._set_padding_lengths_from_model()

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

        if self.test_embeddings:
            if self.pretrained_param is None:
                self.word_params = self.embedding_params.pop('words')
                self.pretrained_param = self.word_params.pop('pretrained_file', None)
                self.use_rand = self.word_params.pop('use_rand', None)

            self.test_dataset = self.load_dataset_from_files(self.test_files)

            # overloaded_param = 1

            # Load embeddings from file
            if self.embeddings is None:
                [self.embeddings, self.dim] = PretrainedEmbeddings.read_embeddings_file(self.pretrained_param)


            embedding_weights = None
            # get embedding layer
            for l in self.model.layers:
                print("Checking",l.name)
                if l.name == 'words_embedding':
                    embedding_weights = l.get_weights()
                    break

            if embedding_weights is None:
                logger.info("Error: can't find embeddings layer")
            else:
                # change weights in layer
                logger.info("%d %d %d %d %s %s", len(embedding_weights),len(embedding_weights[0]),len(embedding_weights[0][0]),self.data_indexer.get_vocab_size(),self.data_indexer.get_word_from_index(3),str(embedding_weights[0][3][:10]))
                self.data_indexer.fit_word_existing_dictionary(self.test_dataset, self.embeddings, self.use_rand)
                logger.info("%d %d %d %d %s %s", len(embedding_weights),len(embedding_weights[0]),len(embedding_weights[0][0]),self.data_indexer.get_vocab_size(),self.data_indexer.get_word_from_index(3),str(embedding_weights[0][3][:10]))

                word_indices = self.data_indexer.words_in_index('words')
                for i in range(2, len(word_indices)):
                    word = self.data_indexer.get_word_from_index(i)

                    if word is None:
                        break

                    if self.use_rand and word not in self.embeddings:
                        vec = numpy.random.rand(self.dim,)
                    else:
                        vec = self.embeddings[word]

                    # if i < 10:
                    #     logger.info("%s %s %s", i,word,str(vec[:10]))

                    embedding_weights[0][i] = vec

                logger.info("Done, now setting weights")
                l.set_weights(embedding_weights)
                logger.info("Done setting weights")

    @overrides
    def _overall_debug_output(self, output_dict: Dict[str, numpy.array]) -> str:
        """
        We'll do something different here: if "embedding" is in output_dict, we'll output the
        embedding matrix at the top of the debug file.  Note that this could be _huge_ - you should
        only do this for debugging on very simple datasets.
        """
        result = super(TextTrainer, self)._overall_debug_output(output_dict)
        if any('embedding' in layer_name for layer_name in output_dict.keys()):
            embedding_layers = set([n for n in output_dict.keys() if 'embedding' in n])
            for embedding_layer in embedding_layers:
                if '_projection' in embedding_layer:
                    continue
                if embedding_layer.startswith('combined_'):
                    continue
                result += self.__render_embedding_matrix(embedding_layer.replace("_embedding", ""))
        return result

    @overrides
    def _uses_data_generators(self):
        return self.data_generator is not None

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = super(TextTrainer, cls)._get_custom_objects()
        custom_objects["TimeDistributedEmbedding"] = TimeDistributedEmbedding
        for value in encoders.values():
            if value.__name__ not in ['LSTM']:
                custom_objects[value.__name__] = value
        for name, layer in TextInstance.tokenizer.get_custom_objects().items():
            custom_objects[name] = layer
        return custom_objects

    #################
    # Utility methods - meant to be called by subclasses, not overriden
    #################

    def _get_sentence_shape(self, sentence_length: int=None) -> Tuple[int]:
        """
        Returns a tuple specifying the shape of a tensor representing a sentence.  This is not
        necessarily just (self.num_sentence_words,), because different text_encodings lead to
        different tensor shapes.  If you have an input that is a sequence of words, you need to
        call this to get the shape to pass to an ``Input`` layer.  If you don't, your model won't
        work correctly for all tokenizers.
        """
        if sentence_length is None:
            # This can't be the default value for the function argument, because
            # self.num_sentence_words will not have been set at class creation time.
            sentence_length = self.num_sentence_words
        return self.tokenizer.get_sentence_shape(sentence_length, self.num_word_characters)

    def _embed_input(self, input_layer: Layer, embedding_suffix: str=""):
        """
        This function embeds a word sequence input, using an embedding defined by
        ``embedding_suffix``.  You should call this function in your ``_build_model`` method any time
        you want to convert word indices into word embeddings.  Note that if this is used in
        conjunction with ``_get_sentence_shape``, we will do the correct thing for whatever
        :class:`~deep_qa.data.tokenizers.tokenizer.Tokenizer` you use.  The actual input to this
        might be words and characters, and we might actually do a concatenation of a word embedding
        and a character-level encoder.  All of this is handled transparently to your concrete model
        subclass, if you use the API correctly, calling ``_get_sentence_shape()`` to get the shape
        for your ``Input`` layer, and passing that input layer into this ``_embed_input()`` method.

        We need to take the input Layer here, instead of just returning a Layer that you can use as
        you wish, because we might have to apply several layers to the input, depending on the
        parameters you specified for embedding things.  So we return, essentially,
        ``embedding(input_layer)``.

        The input layer can have arbitrary shape, as long as it ends with a word sequence.  For
        example, you could pass in a single sentence, a set of sentences, or a set of sets of
        sentences, and we will handle them correctly.

        Internally, we will create a dictionary mapping embedding names to embedding layers, so if
        you have several things you want to embed with the same embedding layer, be sure you use
        the same name each time (or just don't pass a name, which accomplishes the same thing).  If
        for some reason you want to have different embeddings for different inputs, use a different
        name for the embedding.

        In this function, we pass the work off to self.tokenizer, which might need to do some
        additional processing to actually give you a word embedding (e.g., if your text encoder
        uses both words and characters, we need to run the character encoder and concatenate the
        result with a word embedding).

        Note that the ``embedding_suffix`` parameter is a `suffix` to whatever name the tokenizer
        will give to the embeddings it creates.  Typically, the tokenizer will use the name
        ``words``, though it could also use ``characters``, or something else.  So if you pass
        ``_A`` for ``embedding_suffix``, you will end up with actual embedding names like
        ``words_A`` and ``characters_A``.  These are the keys you need to specify in your parameter
        file, for embedding sizes etc.  When constructing actual ``Embedding``
        layers, we will further append the string ``_embedding``, so the layer would be named
        ``words_A_embedding``.
        """
        return self.tokenizer.embed_input(input_layer,
                                          self.__get_embedded_input,
                                          self,
                                          embedding_suffix)

    def _get_encoder(self, name="default", fallback_behavior: str=None):
        """
        This method is intended to be used in your ``_build_model`` implementation, any time you
        want to convert a sequence of vectors into a single vector.  The encoder ``name``
        corresponds to entries in the ``encoder`` parameter passed to the constructor of this
        object, allowing you to customize the kind and behavior of the encoder just through
        parameters.

        A sentence encoder takes as input a sequence of word embeddings, and returns as output a
        single vector encoding the sentence.  This is typically either a simple RNN or an LSTM, but
        could be more complex, if the "sentence" is actually a logical form.

        Parameters
        ----------
        name : str, optional (default="default")
            The name of the encoder.  Multiple calls to ``_get_encoder`` using the same name will
            return the same encoder.  To get parameters for creating the encoder, we look in
            ``self.encoder_params``, which is specified by the ``encoder`` parameter in
            ``self.__init__``.  If ``name`` is not a key in ``self.encoder_params``, the behavior
            is defined by the ``fallback_behavior`` parameter.
        fallback_behavior : str, optional (default=None)
            Determines what to do when ``name`` is not a key in ``self.encoder_params``.  If you
            pass ``None`` (the default), we will use ``self.encoder_fallback_behavior``, specified
            by the ``encoder_fallback_behavior`` parameter to ``self.__init__``.  There are three
            options:

            - ``"crash"``: raise an error.  This is the default for
              ``self.encoder_fallback_behavior``.  The intention is to help you find bugs - if you
              specify a particular encoder name in ``self._build_model`` without giving a fallback
              behavior, you probably wanted to use a particular set of parameters, so we crash if
              they are not provided.
            - ``"use default params"``: In this case, we return a new encoder created with
              ``self.encoder_params["default"]``.
            - ``"use default encoder"``: In this case, we `reuse` the encoder created with
              ``self.encoder_params["default"]``.  This effectively changes the ``name`` parameter
              to ``"default"`` when the given ``name`` is not in ``self.encoder_params``.
        """
        if fallback_behavior is None:
            fallback_behavior = self.encoder_fallback_behavior
        if name in self.encoder_layers:
            # If we've already created this encoder, we can just return it.
            return self.encoder_layers[name]
        if name not in self.encoder_params and name != "default":
            # If we haven't, we need to check that we _can_ create it, and decide _how_ to create
            # it.
            if fallback_behavior == "crash":
                raise ConfigurationError("You asked for a named encoder (" + name + "), but "
                                         "did not provide parameters for that encoder")
            elif fallback_behavior == "use default encoder":
                name = "default"
                params = deepcopy(self.encoder_params.get(name, {}))
            elif fallback_behavior == "use default params":
                params = deepcopy(self.encoder_params["default"])
            else:
                raise ConfigurationError("Unrecognized fallback behavior: " + fallback_behavior)
        else:
            params = deepcopy(self.encoder_params.get(name, {}))
        if name not in self.encoder_layers:
            # We need to check if we've already created this again, because in some cases we change
            # the name in the logic above.
            encoder_layer_name = name + "_encoder"
            new_encoder = self.__get_new_encoder(params, encoder_layer_name)
            self.encoder_layers[name] = new_encoder
        return self.encoder_layers[name]

    def _get_seq2seq_encoder(self, name="default", fallback_behavior: str=None):
        """
        This method is intended to be used in your ``_build_model`` implementation, any time you
        want to convert a sequence of vectors into another sequence of vector.  The encoder
        ``name`` corresponds to entries in the ``encoder`` parameter passed to the constructor of
        this object, allowing you to customize the kind and behavior of the encoder just through
        parameters.

        A seq2seq encoder takes as input a sequence of vectors, and returns as output a sequence of
        vectors.  This method is essentially identical to ``_get_encoder``, except that it gives an
        encoder that returns a sequence of vectors instead of a single vector.

        Parameters
        ----------
        name : str, optional (default="default")
            The name of the encoder.  Multiple calls to ``_get_seq2seq_encoder`` using the same
            name will return the same encoder.  To get parameters for creating the encoder, we look
            in ``self.seq2seq_encoder_params``, which is specified by the ``seq2seq_encoder``
            parameter in ``self.__init__``.  If ``name`` is not a key in
            ``self.seq2seq_encoder_params``, the behavior is defined by the ``fallback_behavior``
            parameter.
        fallback_behavior : str, optional (default=None)
            Determines what to do when ``name`` is not a key in ``self.seq2seq_encoder_params``.
            If you pass ``None`` (the default), we will use
            ``self.seq2seq_encoder_fallback_behavior``, specified by the
            ``seq2seq_encoder_fallback_behavior`` parameter to ``self.__init__``.  There are three
            options:

            - ``"crash"``: raise an error.  This is the default for
              ``self.seq2seq_encoder_fallback_behavior``.  The intention is to help you find bugs -
              if you specify a particular encoder name in ``self._build_model`` without giving a
              fallback behavior, you probably wanted to use a particular set of parameters, so we
              crash if they are not provided.
            - ``"use default params"``: In this case, we return a new encoder created with
              ``self.seq2seq_encoder_params["default"]``.
            - ``"use default encoder"``: In this case, we `reuse` the encoder created with
              ``self.seq2seq_encoder_params["default"]``.  This effectively changes the ``name``
              parameter to ``"default"`` when the given ``name`` is not in
              ``self.seq2seq_encoder_params``.
        """
        if fallback_behavior is None:
            fallback_behavior = self.seq2seq_encoder_fallback_behavior
        if name in self.seq2seq_encoder_layers:
            # If we've already created this encoder, we can just return it.
            return self.seq2seq_encoder_layers[name]
        if name not in self.seq2seq_encoder_params:
            # If we haven't, we need to check that we _can_ create it, and decide _how_ to create
            # it.
            if fallback_behavior == "crash":
                raise ConfigurationError("You asked for a named seq2seq encoder (" + name + "), "
                                         "but did not provide parameters for that encoder")
            elif fallback_behavior == "use default encoder":
                name = "default"
                params = deepcopy(self.seq2seq_encoder_params[name])
            elif fallback_behavior == "use default params":
                params = deepcopy(self.seq2seq_encoder_params["default"])
            else:
                raise ConfigurationError("Unrecognized fallback behavior: " + fallback_behavior)
        else:
            params = deepcopy(self.seq2seq_encoder_params[name])
        if name not in self.seq2seq_encoder_layers:
            # We need to check if we've already created this again, because in some cases we change
            # the name in the logic above.
            encoder_layer_name = name + "_encoder"
            new_encoder = self.__get_new_seq2seq_encoder(params, encoder_layer_name)
            self.seq2seq_encoder_layers[name] = new_encoder
        return self.seq2seq_encoder_layers[name]

    def _set_text_lengths_from_model_input(self, input_slice):
        """
        Given an input slice (a tuple) from a model representing the max length of the sentences
        and the max length of each words, set the padding max lengths.  This gets called when
        loading a model, and is necessary to get padding correct when using loaded models.
        Subclasses need to call this in their ``_set_padding_lengths_from_model`` method.

        Parameters
        ----------
        input_slice : tuple
            A slice from a concrete model class that represents an input word sequence. The tuple
            must be of length one or two, and the first dimension should correspond to the length
            of the sentences while the second dimension (if provided) should correspond to the
            max length of the words in each sentence.
        """
        if len(input_slice) > 2:
            raise ValueError("Length of input tuple must be "
                             "2 or 1, got input tuple of "
                             "length {}".format(len(input_slice)))
        self.num_sentence_words = input_slice[0]
        if len(input_slice) == 2:
            self.num_word_characters = input_slice[1]

    ##################
    # Abstract methods - you MUST override these
    ##################

    def _instance_type(self) -> Instance:
        """
        When reading datasets, what :class:`~deep_qa.data.instances.instance.Instance` type should
        we create?  The ``Instance`` class contains code that creates actual numpy arrays, so this
        instance type determines the inputs that you will get to your model, and the outputs that
        are used for training.
        """
        raise NotImplementedError

    def _set_padding_lengths_from_model(self):
        """
        This gets called when loading a saved model.  It is analogous to ``_set_padding_lengths``,
        but needs to set all of the values set in that method just by inspecting the loaded model.
        If we didn't have this, we would not be able to correctly pad data after loading a model.
        """
        # TODO(matt): I wonder if we can be fancy here and remove this method, instead using
        # `self._instance_type` to figure out what this should be ourselves, or delegating it to
        # the `Instance` type.  But that might run into issues with dynamic padding, though,
        # actually - how can the `Instance` know which things you want your model to pad
        # dynamically?
        raise NotImplementedError

    ########################
    # Model-specific methods - if you do anything complicated, you probably need to override these,
    # but simple models might be able to get by with just the default implementation.  Some of
    # these methods are also callable by non-TextTrainer objects, so that we can separate out the
    # DataGenerator and other functionality.
    ########################

    def get_instance_sorting_keys(self) -> List[str]:  # pylint: disable=no-self-use
        """
        If we're using dynamic padding, we want to group the instances by padding length, so that
        we minimize the amount of padding necessary per batch.  This variable sets what exactly
        gets sorted by.  We'll call
        :func:`~deep_qa.data.instances.instance.IndexedInstance.get_padding_lengths()` on each
        instance, pull out these keys, and sort by them in the order specified.  You'll want to
        override this in your model class if you have more complex models.

        The default implementation is to sort first by ``num_sentence_words``, then by
        ``num_word_characters`` (if applicable).
        """
        sorting_keys = ['num_sentence_words']
        if isinstance(self.tokenizer, tokenizers['words and characters']):
            # TODO(matt): This is a bit brittle, because other tokenizers might need similar
            # handling.  We could consider adding an API call to Tokenizer classes to get this kind
            # of information.  If we find ourselves adding more tokenizers, it might be worth it.
            sorting_keys.append('num_word_characters')
        return sorting_keys

    def get_padding_lengths(self) -> Dict[str, int]:
        """
        This is about padding.  Any solver will have some number of things that need padding in
        order to make consistently-sized data arrays, like the length of a sentence.  This method
        returns a dictionary of all of those things, mapping a length key to an int.

        If any of the entries in this dictionary is ``None``, the padding code will calculate a
        padding length from the data itself.  This could either be a good idea or a bad idea - if
        you have outliers in your data, you could be wasting a whole lot of memory and computation
        time if you pad the whole dataset to the size of the outlier.  On the other hand, if you do
        batch-specific padding, this can save you a whole lot of time, if you group batches by
        similar lengths.

        Here we return the lengths that are applicable to encoding words and sentences.  If you
        have additional padding dimensions, call super().get_padding_lengths() and then update the
        dictionary.
        """
        return self.tokenizer.get_padding_lengths(self.num_sentence_words, self.num_word_characters)

    def _set_padding_lengths(self, dataset_padding_lengths: Dict[str, int]):
        """
        This is about padding.  Any model will have some number of things that need padding in
        order to make a consistent set of input arrays, like the length of a sentence.  This method
        sets those variables given a dictionary of lengths from a dataset.

        Note that you might choose not to update some of these lengths, either because you want to
        keep the model flexible to allow for dynamic (batch-specific) padding, or because you've
        set a hard limit in the class parameters and don't want to change it.
        """
        if self.data_generator is not None and self.data_generator.dynamic_padding:
            return
        if self.num_sentence_words is None:
            self.num_sentence_words = dataset_padding_lengths.get('num_sentence_words', None)
        if self.num_word_characters is None:
            self.num_word_characters = dataset_padding_lengths.get('num_word_characters', None)

    # pylint: disable=no-self-use,unused-argument
    def get_padding_memory_scaling(self, padding_lengths: Dict[str, int]) -> int:
        """
        This method is for computing adaptive batch sizes.  We assume that memory usage is a
        function that looks like this: :math:`M = b * O(p) * c`, where :math:`M` is the memory
        usage, :math:`b` is the batch size, :math:`c` is some constant that depends on how much GPU
        memory you have and various model hyperparameters, and :math:`O(p)` is a function outlining
        how memory usage asymptotically varies with the padding lengths.  Our approach will be to
        let the user effectively set :math:`\\frac{M}{c}` using the
        ``adaptive_memory_usage_constant`` parameter in :class:`DataGenerator`.  The model (this
        method) specifies :math:`O(p)`, so we can solve for the batch size :math:`b`.  The more
        specific you get in specifying :math:`O(p)` in this function, the better a job we can do in
        optimizing memory usage.

        Parameters
        ----------
        padding_lengths: Dict[str, int]
            Dictionary containing padding lengths, mapping keys like ``num_sentence_words`` to
            ints.  This method computes a function of these ints.

        Returns
        -------
        O(p): int
            The big-O complexity of the model, evaluated with the specific ints given in
            ``padding_lengths`` dictionary.
        """
        # This is a RuntimeError instead of a NotImplementedError because it's not required to
        # implement this method to have a valid TextTrainer.  You only have to implement it if you
        # want to use adaptive batch sizes.
        raise RuntimeError("You need to implement this method for your model!")
    # pylint: enable=no-self-use,unused-argument

    #################
    # Private methods - you can't to override these.  If you find yourself needing to, we can
    # consider making them protected instead.
    #################

    def __get_embedded_input(self,
                             input_layer: Layer,
                             embedding_name: str,
                             vocab_name: str='words'):
        """
        This function does most of the work for self._embed_input.  We pass this method to the
        tokenizer, so it can get whatever embedding layers it needs.

        We allow for multiple vocabularies, e.g., if you want to embed both characters and words
        with separate embedding matrices.
        """
        if embedding_name not in self.embedding_layers:
            self.embedding_layers[embedding_name] = self.__get_new_embedding(embedding_name, vocab_name)

        embedding_layer, projection_layer, dropout = self.embedding_layers[embedding_name]
        embedded_input = embedding_layer(input_layer)

        layer_name = clean_layer_name(input_layer.name, strip_numerics_after_underscores=False)
        if projection_layer is not None:
            # 1 here to account for batch_size, which we don't need
            # to TimeDistribute.
            for i in range(1, K.ndim(input_layer)):
                projection_layer_name = layer_name + "/" + projection_layer.name + "_{}".format(i)
                projection_layer = TimeDistributed(projection_layer, name=projection_layer_name)
            embedded_input = projection_layer(embedded_input)
        if dropout > 0.0:
            embedded_input = Dropout(dropout)(embedded_input)

        return embedded_input

    def __get_new_embedding(self, name: str, vocab_name: str='words'):
        """
        Creates an Embedding Layer (and possibly also a Dense projection Layer) based on the
        parameters you've passed to the TextTrainer.  These could be pre-trained embeddings or not,
        could include a projection or not, and so on.

        Parameters
        ----------
        name : ``str``
            The name of the embedding.  This needs to correspond to one of the keys in the
            ``embeddings`` parameter dictionary passed to the constructor.
        """
        embedding_params = self.embedding_params.pop(name)
        with tensorflow.device("/cpu:0"):
            pretrained_file = embedding_params.pop('pretrained_file', None)
            projection_layer = None
            if pretrained_file:
                embedding_layer = PretrainedEmbeddings.get_embedding_layer(
                        pretrained_file,
                        self.data_indexer,
                        embedding_params.pop('fine_tune', False),
                        name=name + '_embedding')

                if embedding_params.pop('project', False):
                    # This projection layer is not time distributed, because we handle it later
                    # in __get_embedded_input - this allows us to more easily reuse embeddings
                    # for inputs with different shapes, as Keras sets layer attributes such as
                    # input shape the first time the layer is called, which is overly restrictive
                    # in the case of sharing embedding lookup tables.
                    projection_layer = Dense(units=embedding_params.pop('dimension'), name=name + "_projection")
                else:
                    embedding_dimension = embedding_params.pop('dimension', None)
                    if embedding_dimension is not None and embedding_dimension != embedding_layer.output_dim:
                        raise ConfigurationError("You have specified both 'pretrained_file' "
                                                 " and 'dimension' in your embedding parameters, but "
                                                 "the 'project' argument was either False or unset and the "
                                                 "dimension you specified was not equal to the pretrained"
                                                 " embedding size. Refusing to continue without clarification"
                                                 " of parameters.")
            else:
                # TimeDistributedEmbedding works with inputs of any shape.
                embedding_layer = TimeDistributedEmbedding(
                        input_dim=self.data_indexer.get_vocab_size(vocab_name),
                        output_dim=embedding_params.pop('dimension'),
                        mask_zero=True,  # this handles padding correctly
                        name=name + '_embedding')
                if embedding_params.pop('project', False):
                    raise ConfigurationError("You are projecting randomly initialised embeddings. Change "
                                             " 'project' to false or add pretrained_file to your config. ")
            dropout = embedding_params.pop('dropout', 0.5)

            # We now should have popped all parameters from this
            # embedding scope, so we check for any which remain.
            embedding_params.assert_empty("embedding with name {}".format(name))
            return embedding_layer, projection_layer, dropout

    def __get_new_encoder(self, params: Params, name: str):
        encoder_type = params.pop_choice("type", list(encoders.keys()),
                                         default_to_first_choice=True)
        params["name"] = name
        params.setdefault("units", self.embedding_layers['words'][0].output_dim)
        set_regularization_params(encoder_type, params)
        return encoders[encoder_type](**params)

    def __get_new_seq2seq_encoder(self, params: Params, name="seq2seq_encoder"):
        encoder_params = params["encoder_params"]
        wrapper_params = params["wrapper_params"]
        wrapper_params["name"] = name
        seq2seq_encoder_type = encoder_params.pop_choice("type", list(seq2seq_encoders.keys()),
                                                         default_to_first_choice=True)
        encoder_params.setdefault("units", self.embedding_layers['words'][0].output_dim)
        set_regularization_params(seq2seq_encoder_type, encoder_params)
        return seq2seq_encoders[seq2seq_encoder_type](**params)

    def __render_embedding_matrix(self, embedding_name: str) -> str:
        result = 'Embedding matrix for %s:\n' % embedding_name
        embedding_weights = self.embedding_layers[embedding_name][0].get_weights()[0]
        for i in range(self.data_indexer.get_vocab_size()):
            word = self.data_indexer.get_word_from_index(i)
            word_vector = '[' + ' '.join('%.4f' % x for x in embedding_weights[i]) + ']'
            result += '%s\t%s\n' % (word, word_vector)
        result += '\n'
        return result
