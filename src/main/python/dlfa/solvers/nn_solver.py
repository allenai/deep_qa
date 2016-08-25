import sys
import pickle

from typing import List

import numpy
from keras.layers import Dense, Input, Embedding, TimeDistributed, Dropout
from keras.models import Model, model_from_json

from ..data.dataset import TextDataset, IndexedDataset  # pylint: disable=unused-import
from ..data.embeddings import PretrainedEmbeddings
from ..data.index_data import DataIndexer


class NNSolver(object):
    def __init__(self, **kwargs):
        """
        Allowed kwargs are specified in update_arg_parser()
        """
        # Note that because of how we called vars() on the parsed arguments, the defaults specified
        # in update_arg_parser() will already be present here, including None values, so we can
        # just grab these directly without worrying if they're there or not.
        self.model_prefix = kwargs['model_serialization_prefix']

        self.pretrained_embeddings_file = kwargs['pretrained_embeddings_file']
        self.fine_tune_embeddings = kwargs['fine_tune_embeddings']
        self.project_embeddings = kwargs['project_embeddings']
        self.embedding_size = kwargs['embedding_size']
        self.embedding_dropout = kwargs['embedding_dropout']
        self.max_sentence_length = kwargs['max_sentence_length']
        self.max_training_instances = kwargs['max_training_instances']

        self.train_file = kwargs['train_file']
        self.positive_train_file = kwargs['positive_train_file']
        self.negative_train_file = kwargs['negative_train_file']
        self.validation_file = kwargs['validation_file']
        self.test_file = kwargs['test_file']
        self.debug_file = kwargs['debug_file']

        self.num_epochs = kwargs['num_epochs']
        self.patience = kwargs['patience']

        self.data_indexer = DataIndexer()
        self.model = None
        self.debug_model = None
        self.best_epoch = -1
        self.embedding_layer = None
        self.time_distributed_embedding_layer = None
        self.projection = None
        self.time_distributed_projection = None

    @classmethod
    def update_arg_parser(cls, parser):
        """
        MODEL SPECIFICATION:
            embedding_size: int. Size of word vectors (default 50).
            pretrained_embeddings_file: str.
            max_sentence_length: max length of training sentences (ignored at test time).

        DATA SPECIFICATION:
            train_file: path to training data.
            positive_train_file: path to positive training data.
            negative_train_file: path to negative training data.
            validation_file: path to validation data.

            NOTE on train file arguments: if `train_file` is given, the other two arguments are
            ignored, and the file is assumed to have instance labels.  If `positive_train_file` is
            given, it is assumed to not have labels (or all labels must be "1").  Similarly for
            `negative_train_file`, except label must be "0" if present.  If `positive_train_file` is
            given and `negative_train_file` isn't, we will try to generate negative data, but the
            method to do so is poor and won't work for all subclasses.

        TRAINING HYPER-PARAMETERS:
            max_training_instances: if this is given and we have more training instances than this,
                trunction them.
            num_epochs: how many epochs to train for (default 20).
            patience: number of epochs to be patient before stopping early (default 1).
        """
        # TODO(matt): move comments in the docstring into help text, so we're not repeating
        # ourselves.

        # Input files
        parser.add_argument('--train_file', type=str)
        parser.add_argument('--positive_train_file', type=str)
        parser.add_argument('--negative_train_file', type=str)
        parser.add_argument('--validation_file', type=str)
        parser.add_argument('--test_file', type=str)
        parser.add_argument('--debug_file', type=str,
                            help='Visualize the intermediate outputs of the trained model.'
                                 'Output will be written to <model_serialization_prefix>_debug_<epoch>.txt')

        # Model specification
        parser.add_argument("--model_serialization_prefix", required=True,
                            help="Prefix for saving and loading model files")
        parser.add_argument('--embedding_size', type=int, default=50,
                            help="Number of dimensions to use for word embeddings")
        parser.add_argument('--pretrained_embeddings_file', type=str,
                            help="If specified, we will use the vectors in this file and learn a "
                            "projection matrix to get word vectors of dimension `embedding_size`, "
                            "instead of learning the embedding matrix ourselves.")
        parser.add_argument('--fine_tune_embeddings', action='store_true',
                            help="If we're using pre-trained embeddings, should we fine tune them?")
        parser.add_argument('--project_embeddings', action='store_true',
                            help="Should we have a projection layer on top of our embedding layer? "
                            "(mostly useful with pre-trained embeddings)")
        parser.add_argument('--embedding_dropout', type=float, default=0.5,
                            help="Dropout parameter to apply to the word embedding layer")
        parser.add_argument('--max_sentence_length', type=int,
                            help="Upper limit on length of training data. Ignored during testing.")

        # Training details
        parser.add_argument('--max_training_instances', type=int,
                            help="Upper limit on the size of training data")
        parser.add_argument('--num_epochs', type=int, default=20,
                            help="Number of train epochs (20 by default)")
        parser.add_argument('--patience', type=int, default=1,
                            help="Number of epochs to be patient before early stopping (1 by default)")

        # Testing details
        parser.add_argument('--use_model_from_epoch', type=int,
                            help="Use model from a particular epoch (use best saved model if empty)")

    def prep_labeled_data(self, dataset: TextDataset, for_train: bool):
        """
        Takes dataset, which could be a complex tuple for some classes, and produces as output a
        tuple of (inputs, labels), which can be used directly with Keras to either train or
        evaluate self.model.

        For training and validation data, this method is called internally during self.train().  If
        you want to evaluate the model on some other test dataset, this is the method you need to
        call.  However, that dataset has to have labels, or this method will crash.  We don't
        currently have an API for making predictions on data that doesn't have labels.  TODO(matt)
        """
        raise NotImplementedError

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

    def train(self):
        '''
        Trains the model.

        All training parameters have already been passed to the constructor, so we need no
        arguments to this method.
        '''

        # First we need to prepare the data that we'll use for training.
        train_input, train_labels = self._get_training_data()
        validation_input, validation_labels = self._get_validation_data()

        # Then we build the model and compile it.
        self.model = self._build_model(train_input)
        # TODO(pradeep): Try out other optimizers, especially rmsprop.
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if self.debug_file:
            # Get the list of layers whose outputs will be visualized as per the
            # solver definition and build a debug model.
            debug_layers = self.get_debug_layer_names()
            self.debug_model = self._build_debug_model(debug_layers)
            self.debug_model.compile(loss='mse', optimizer='sgd')  # Will not train this model.
            debug_dataset, debug_input = self._get_debug_dataset_and_input()
            do_debug = True
        else:
            do_debug = False

        # Now we actually train the model, with patient early stopping using the validation data.
        best_accuracy = 0.0
        self.best_epoch = 0
        num_worse_epochs = 0
        for epoch_id in range(self.num_epochs):
            print("Epoch %d" % epoch_id, file=sys.stderr)
            self.model.fit(train_input, train_labels, nb_epoch=1)
            accuracy = self.evaluate(validation_labels, validation_input)
            print("Validation accuracy: %.4f" % accuracy, file=sys.stderr)
            if accuracy < best_accuracy:
                num_worse_epochs += 1
                if num_worse_epochs >= self.patience:
                    print("Stopping training", file=sys.stderr)
                    break
            else:
                best_accuracy = accuracy
                self.best_epoch = epoch_id
                num_worse_epochs = 0  # Reset the counter.
                self._save_model(epoch_id)
                if do_debug:
                    # Shows intermediate outputs of the model on validation data
                    self.debug(debug_dataset, debug_input, epoch_id)
        self._save_best_model()

    def test(self):
        """
        Tests the model, using the file given to the constructor.
        """
        inputs, labels = self._get_test_data()
        print("Scoring test data", file=sys.stderr)
        accuracy = self.evaluate(labels, inputs)
        print("Test accuracy: %.4f" % accuracy, file=sys.stderr)

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
        # Loading serialized model
        model_config_file = open("%s_config.json" % self.model_prefix)
        model_config_json = model_config_file.read()
        self.model = model_from_json(model_config_json,
                                     custom_objects=self._get_custom_objects())
        if epoch is not None:
            model_file = "%s_weights_epoch=%d.h5" % (self.model_prefix, epoch)
        else:
            model_file = "%s_weights.h5" % self.model_prefix
        self.model.load_weights(model_file)
        data_indexer_file = open("%s_data_indexer.pkl" % self.model_prefix, "rb")
        self.data_indexer = pickle.load(data_indexer_file)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        model_config_file.close()
        data_indexer_file.close()
        self._set_max_lengths_from_model()

    def score(self, test_input):
        return self.model.predict(test_input)

    @staticmethod
    def _assert_dataset_is_questions(dataset: TextDataset):
        """
        This method checks that dataset matches the assumptions we make about validation data: that
        it is a list of sentences corresponding to four-choice questions, with one correct answer
        for every four instances.

        So, specifically, we check that the number of instances is a multiple of four, and we check
        that each group of four instances has exactly one instance with label True, and all other
        labels are False (i.e., no None labels for validation data).
        """
        assert len(dataset.instances) % 4 == 0, "Not enough lines per question"
        questions = zip(*[dataset.instances[i::4] for i in range(4)])
        for question in questions:
            question_labels = [instance.label for instance in question]
            label_counts = {x: question_labels.count(x) for x in set(question_labels)}
            assert label_counts[True] == 1, "Must have one correct answer option"
            assert label_counts[False] == 3, "Must have three incorrect answer options"

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

        We assume that the data that produced `labels` has already been validated with
        NNSolver._assert_dataset_is_questions(), so we do not do any checking here.  See the comments
        there for the requirements on the input data.
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

    def _set_max_lengths_from_model(self):
        """
        Given a loaded model, set the max_lengths needed for padding.  This is necessary so that we
        can pad the test data if we just loaded a saved model.
        """
        raise NotImplementedError

    def _get_training_data(self):
        """Loads training data and converts it into a format suitable for input to Keras.  This
        method must return a tuple of (train_input, train_labels).

        This method takes no arguments; any necessary arguments (e.g., a path for where to find the
        training data) must have been passed to the constructor of this object.

        This base implementation is suitable for NNSolvers that only take single sentences /
        logical forms as input.  NNSolvers that have more complicated inputs will need to override
        this method.
        """
        if self.train_file:
            dataset = TextDataset.read_from_file(self.train_file)
        else:
            positive_dataset = TextDataset.read_from_file(self.positive_train_file, label=True)
            negative_dataset = TextDataset.read_from_file(self.negative_train_file, label=False)
            dataset = positive_dataset.merge(negative_dataset)
        if self.max_training_instances is not None:
            print("Truncating the dataset to", self.max_training_instances, "instances")
            dataset = dataset.truncate(self.max_training_instances)
        self.data_indexer.fit_word_dictionary(dataset)
        return self.prep_labeled_data(dataset, for_train=True)

    def _get_validation_data(self):
        """
        Like _get_training_data, but for validation data.  Also, while _get_training_data() returns
        instance-level train_labels, for computing gradients during training, for validation we
        group the labels by question, to use question accuracy as our early stopping criterion.

        This base implementation is suitable for NNSolvers that only take single sentences /
        logical forms as input.  NNSolvers that have more complicated inputs will need to override
        this method.
        """
        return self._read_question_data(self.validation_file)

    def _get_test_data(self):
        return self._read_question_data(self.test_file)

    def _read_question_data(self, filename):
        dataset = TextDataset.read_from_file(filename)
        self._assert_dataset_is_questions(dataset)
        inputs, labels = self.prep_labeled_data(dataset, for_train=False)
        return inputs, self.group_by_question(labels)

    def _index_and_pad_dataset(self, dataset: TextDataset, max_lengths: List[int]):
        indexed_dataset = dataset.to_indexed_dataset(self.data_indexer)
        indexed_dataset.pad_instances(max_lengths)
        return indexed_dataset

    def _build_model(self, train_input):
        """Constructs and returns a Keras model that will take train_input as input, and produce as
        output a true/false decision for each input.

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

    def _get_embedded_sentence_input(self, sentence_input, is_time_distributed=False):
        """
        Performs the initial steps of embedding sentences.  This function takes care of two steps:
        (1) it creates an Input layer for the sentence input, and (2) it converts word indices into
        word vectors with an Embedding layer, if necessary.  If dropout has been specified on the
        embedding layer, we also apply that here.

        Because Keras requires access to the actual Input() layer, we return that along with the
        final word vector layer.
        """
        # pylint: disable=redefined-variable-type
        input_layer = Input(shape=sentence_input.shape[1:], dtype='int32')

        # If a particular model has several places where sentences are encoded, we want to be sure
        # to use the _same_ embedding layer for all of them, so we initialize them and save them to
        # self here, if we haven't already done so.
        if self.embedding_layer is None:
            if self.pretrained_embeddings_file:
                # If we have a pre-trained embeddings file, we'll load a fixed Embedding layer,
                # then add a projection on top of it.  So our "embedding layer" is actually two
                # layers: a non-trainable Embedding layer, then a TimeDistributed(Dense) layer.
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
                self.projection = TimeDistributed(Dense(output_dim=self.embedding_size),
                                                  name='embedding_projection')
                self.time_distributed_projection = TimeDistributed(self.projection,
                                                                   name='background_embedding_projection')
            self.time_distributed_embedding_layer = TimeDistributed(self.embedding_layer)

        # Now we actually embed the input and apply dropout.
        if is_time_distributed:
            embedded_input = self.time_distributed_embedding_layer(input_layer)
        else:
            embedded_input = self.embedding_layer(input_layer)

        if self.project_embeddings:
            if is_time_distributed:
                embedded_input = self.time_distributed_projection(embedded_input)
            else:
                embedded_input = self.projection(embedded_input)

        if self.embedding_dropout > 0.0:
            embedded_input = Dropout(self.embedding_dropout)(embedded_input)

        return input_layer, embedded_input

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
        return {}
