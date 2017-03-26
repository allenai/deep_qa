import logging
import sys
import pickle
import codecs
import argparse
import random
from nltk.tokenize import word_tokenize

# TODO(matt): we need to refactor this to match the rest of the code, moving the main() method out
# of here and using relative imports.  And we should be using deep_qa.data.data_indexer, anyway.
from index_data import DataIndexer  # pylint: disable=import-error
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Embedding, Dropout, merge, TimeDistributed, Dense, SimpleRNN
from keras.callbacks import EarlyStopping

random.seed(13370)
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class WordReplacer:
    def __init__(self):
        self.data_indexer = DataIndexer()
        self.model = None

    # TODO(matt): factor_base and tokenize should not have default values.  get_substitutes and
    # score_sentences would currently crash if factor_base is trained with anything but 2, because
    # they don't pass in a factor_base.
    def process_data(self, sentences, is_training, max_length=None, factor_base=2, tokenize=True):
        sentence_lengths, indexed_sentences = self.data_indexer.index_data(
                sentences, max_length, tokenize, is_training)
        # We want the inputs to be words 0..n-1 and targets to be words 1..n in all
        # sentences, so that at each time step t, p(w_{t+1} | w_{0}..w_{t}) will be
        # predicted.
        input_array = indexed_sentences[:, :-1] # Removing the last word from each sentence
        target_array = indexed_sentences[:, 1:] # Removing the first word from each sent.
        factored_target_arrays = self.data_indexer.factor_target_indices(target_array, base=factor_base)
        return sentence_lengths, input_array, factored_target_arrays

    def train_model(self, sentences, word_dim=50, factor_base=2, num_epochs=20,
                    tokenize=True, use_lstm=False):
        _, input_array, factored_target_arrays = self.process_data(
                sentences, is_training=True, factor_base=factor_base, tokenize=tokenize)
        vocab_size = self.data_indexer.get_vocab_size()
        num_factors = len(factored_target_arrays)
        model_input = Input(shape=input_array.shape[1:], dtype='int32') # (batch_size, num_words)
        embedding = Embedding(input_dim=vocab_size, output_dim=word_dim, mask_zero=True)
        embedded_input = embedding(model_input) # (batch_size, num_words, word_dim)
        regularized_embedded_input = Dropout(0.5)(embedded_input)

        # Bidirectional RNNs = Two RNNs, with the second one processing the input
        # backwards, and the two outputs concatenated.
        # Return sequences returns output at every timestep, instead of just the last one.
        rnn_model = LSTM if use_lstm else SimpleRNN

        # Since we will project the output of the lstm down to factor_base in the next
        # step anyway, it is okay to project it down a bit now. So output_dim = word_dim/2
        # This minimizes the number of parameters significantly. All four Ws in LSTM
        # will now be half as big as they would be if output_dim = word_dim.
        forward_rnn = rnn_model(output_dim=int(word_dim/2), return_sequences=True, name='forward_rnn')
        backward_rnn = rnn_model(output_dim=int(word_dim/2), go_backwards=True,
                                 return_sequences=True, name='backward_rnn')
        forward_rnn_out = forward_rnn(regularized_embedded_input)  # (batch_size, num_words, word_dim/2)
        backward_rnn_out = backward_rnn(regularized_embedded_input)  # (batch_size, num_words, word_dim/2)
        bidirectional_rnn_out = merge([forward_rnn_out, backward_rnn_out],
                                      mode='concat')  # (batch_size, num_words, word_dim)
        regularized_rnn_out = Dropout(0.2)(bidirectional_rnn_out)

        model_outputs = []
        # Make as many output layers as there are factored target arrays, and the same size
        for i in range(num_factors):
            # TimeDistributed(layer) makes layer accept an additional time distribution.
            # i.e. if layer takes a n-dimensional input, TimeDistributed(layer) takes a
            # n+1 dimensional input, where the second dimension is time (or words in the
            # sentence). We need this now because RNN above returns one output per timestep
            factor_output = TimeDistributed(Dense(units=factor_base,
                                                  activation='softmax',
                                                  name='factor_output_%d' % i))
            model_outputs.append(factor_output(regularized_rnn_out))  # (batch_size, num_words, factor_base)

        # We have num_factors number of outputs in the model. So, the effective output shape is
        # [(batch_size, num_words, factor_base)] * num_factors
        model = Model(inputs=model_input, outputs=model_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        model.summary()
        early_stopping = EarlyStopping()
        model.fit(input_array, factored_target_arrays, nb_epoch=num_epochs, validation_split=0.1,
                  callbacks=[early_stopping])
        self.model = model

    def save_model(self, model_serialization_prefix):
        data_indexer_pickle_file = open("%s_di.pkl" % model_serialization_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_pickle_file)
        model_config = self.model.to_json()
        model_config_file = open("%s_config.json" % model_serialization_prefix, "w")
        print(model_config, file=model_config_file)
        self.model.save_weights("%s_weights.h5" % model_serialization_prefix, overwrite=True)

    def get_model_input_shape(self):
        return self.model.get_input_shape_at(0)

    def load_model(self, model_serialization_prefix):
        self.data_indexer = pickle.load(open("%s_di.pkl" % model_serialization_prefix, "rb"))
        self.model = model_from_json(open("%s_config.json" % model_serialization_prefix).read())
        self.model.load_weights("%s_weights.h5" % model_serialization_prefix)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.model.summary()

    def get_substitutes(self, sentences, locations, train_sequence_length, num_substitutes=5,
                        tokenize=True, search_space_size=5000):
        '''
        sentences (list(str)): List of sentences with words that need to be substituted
        locations (list(int)): List of indices, the same size as sentences, containing
            indices of words in sentences that need to be substituted
        train_sequence_length (int): Length of sequences the model was trained on
        '''
        max_train_length = train_sequence_length + 1  # + 1 because the last word would be stripped
        sentence_lengths, indexed_sentences, _ = self.process_data(
                sentences, is_training=False, max_length=max_train_length, tokenize=tokenize)
        # All prediction factors shape: [(batch_size, num_words, factor_base)] * num_factors
        all_prediction_factors = self.model.predict(indexed_sentences)

        all_substitutes = []
        for sentence_id, (sentence_length, location) in enumerate(zip(sentence_lengths, locations)):
            # If sentence length is greater than the longest sentence seen during training,
            # data indexer will truncate it anyway. So, let's not make expect the predictions
            # to be longer than that.
            sentence_length = min(sentence_length, max_train_length)
            prediction_length = sentence_length - 1  # Ignore the starting <s> symbol

            # Each prediction factor is of the shape
            # (num_sentences, padding_length+num_words, factor_base)
            # We need to take the probability of word given by "location" in sentence
            # given by sentence_id. For that, we need to collect the probabilities over
            # all factors, remove padding, and then look up the factor probabilities at
            # index=location
            word_predictions = [predictions[sentence_id][-prediction_length:][location]
                                for predictions in all_prediction_factors]
            sorted_substitutes = self.data_indexer.unfactor_probabilities(word_predictions, search_space_size)
            all_substitutes.append(sorted_substitutes[:num_substitutes])
        return all_substitutes

    def score_sentences(self, sentences, train_sequence_length, tokenize=True):
        """
        Assigns each input sentence a score using self.model.evaluate().
        """
        max_train_length = train_sequence_length + 1  # + 1 because the last word would be stripped
        scores = []
        for sentence in sentences:
            _, indexed_sentence, factored_target_array = self.process_data(
                    [sentence], is_training=False, max_length=max_train_length, tokenize=tokenize)
            score_and_metrics = self.model.evaluate(indexed_sentence, factored_target_array, verbose=0)
            scores.append(score_and_metrics[0])
        return list(zip(scores, sentences))


def train(train_file, max_instances, factor_base, word_dim, num_epochs, tokenize, use_lstm,
          model_serialization_prefix):
    word_replacer = WordReplacer()
    logger.info("Reading training data")
    train_lines = [x.strip() for x in codecs.open(train_file, "r", "utf-8")]
    if '\t' in train_lines[0]:
        train_sentences = [x.split('\t')[1] for x in train_lines]
    else:
        train_sentences = train_lines
    random.shuffle(train_sentences)
    if max_instances is not None:
        train_sentences = train_sentences[:max_instances]
    word_replacer.train_model(train_sentences, factor_base=factor_base,
                              word_dim=word_dim, num_epochs=num_epochs,
                              tokenize=tokenize, use_lstm=use_lstm)
    word_replacer.save_model(model_serialization_prefix)
    return word_replacer


def corrupt_sentences(word_replacer, file_to_corrupt, search_space_size, tokenize,
                      create_sentence_indices, output_file):
    logger.info("Reading test data")
    test_lines = [x.strip() for x in codecs.open(file_to_corrupt, "r", "utf-8")]
    if '\t' in test_lines[0]:
        test_sentence_strings = [x.split('\t')[1] for x in test_lines]
    else:
        test_sentence_strings = test_lines
    test_sentence_words = [word_tokenize(x) for x in test_sentence_strings]
    test_sentences = []
    locations = []
    # Stop words. Do not replace these or let them be replacements.
    words_to_ignore = set(["<s>", "</s>", "PADDING", ".", ",", "of", "in", "by", "the", "to",
                           "and", "is", "a"])
    for words in test_sentence_words:
        if len(set(words).difference(words_to_ignore)) == 0:
            # This means that there are no non-stop words in the input. Ignore it.
            continue
        # Generate a random location, between 0 and the second last position
        # because the last position is usually a period
        location = random.randint(0, len(words) - 2)
        while words[location] in words_to_ignore:
            location = random.randint(0, len(words) - 2)
        locations.append(location)
        test_sentences.append(" ".join(words))
    train_sequence_length = word_replacer.get_model_input_shape()[1]
    logger.info("Limiting search space size to %d", search_space_size)
    substitutes = word_replacer.get_substitutes(test_sentences, locations,
                                                train_sequence_length, tokenize=tokenize,
                                                search_space_size=search_space_size)
    outfile = codecs.open(output_file, "w", "utf-8")
    index = 0
    for logprob_substitute_list, words, location in zip(substitutes, test_sentence_words, locations):
        word_being_replaced = words[location]
        for _, substitute in logprob_substitute_list:
            if substitute not in set(list(words_to_ignore) + [word_being_replaced]):
                corrupted_words = list(words)
                corrupted_words[location] = substitute
                corrupted_sentence = " ".join(corrupted_words)
                if create_sentence_indices:
                    output_string = '%d\t%s' % (index, corrupted_sentence)
                else:
                    output_string = corrupted_sentence
                print(output_string, file=outfile)
                index += 1
                break
    outfile.close()


def select_mostly_likely_candidates(word_replacer: WordReplacer,
                                    candidates_file: str,
                                    top_k: int,
                                    tokenize: bool,
                                    create_sentence_indices: bool,
                                    max_output_sentences: int,
                                    output_file: str):
    train_sequence_length = word_replacer.get_model_input_shape()[1]
    kept_sentences = []
    logger.info("Selecting most likely candidates")
    index = 0
    for line in codecs.open(candidates_file, "r", "utf-8"):
        index += 1
        if index % 10000 == 0:
            logger.info(index)
        candidates = line.strip().split("\t")
        candidate_scores = word_replacer.score_sentences(candidates, train_sequence_length, tokenize)
        candidate_scores.sort(reverse=True)
        for _, candidate in candidate_scores[:top_k]:
            kept_sentences.append(candidate)
    random.shuffle(kept_sentences)
    if max_output_sentences:
        kept_sentences = kept_sentences[:max_output_sentences]
    with codecs.open(output_file, "w", "utf-8") as outfile:
        index = 0
        for sentence in kept_sentences:
            if create_sentence_indices:
                output_string = '%d\t%s\n' % (index, sentence)
            else:
                output_string = '%s\n' % sentence
            index += 1
            outfile.write(output_string)


def generate_multiple_choice_questions(word_replacer: WordReplacer,
                                       sentences_file: str,
                                       candidates_file: str,
                                       tokenize: bool,
                                       num_false_options: int,
                                       create_sentence_indices: bool,
                                       max_output_sentences: int,
                                       output_file: str):
    train_sequence_length = word_replacer.get_model_input_shape()[1]
    logger.info("Reading sentences file")
    sentences = {}
    for line in codecs.open(sentences_file, "r", "utf-8"):
        (sentence_index, sentence) = line.strip().split("\t")
        sentences[int(sentence_index)] = sentence
    generated_questions = []
    index = 0
    for line in codecs.open(candidates_file, "r", "utf-8"):
        index += 1
        if index % 10 == 0:
            logger.info(index)
        if max_output_sentences is not None and index > max_output_sentences:
            break
        fields = line.strip().split("\t")
        original_sentence_index = int(fields[0])
        candidates = fields[1:]
        candidate_scores = word_replacer.score_sentences(candidates, train_sequence_length, tokenize)
        candidate_scores.sort(reverse=True)
        options = [(sentences[original_sentence_index], True)]
        for _, candidate in candidate_scores[:num_false_options]:
            options.append((candidate, False))
        random.shuffle(options)
        generated_questions.append(options)
    random.shuffle(generated_questions)
    with codecs.open(output_file, "w", "utf-8") as outfile:
        index = 0
        for question in generated_questions:
            for (sentence, label) in question:
                label_string = '1' if label else '0'
                if create_sentence_indices:
                    output_string = '%d\t%s\t%s\n' % (index, sentence, label_string)
                else:
                    output_string = '%s\t%s\n' % (sentence, label_string)
                index += 1
                outfile.write(output_string)


def main():
    description = """Trains and tests a language model using Keras.

    There are currently three kinds of operations performed by this code:
        (1) train a language model using input sentences
        (2) randomly corrupt sentences using the language model to propose replacement words
        (3) use a language model to select the most likely sentence out of several candidates

    To do (1), input the --train_file argument, along with all relevant training parameters.

    To do (2), use the --file_to_corrupt option.

    To do (3), use the --candidates_file option.

    These are not mutually exclusive; you could train a model, then use it to do either (2) or (3),
    by providing two of the above command-line arguments (passing all three would result in writing
    two different outputs to the same file, currently...).

    If you try to use a language model without supplying a training input, we will try to load one
    using the --model_serialization_prefix option.

    The help text for the arguments below are all prefixed with the task number they are applicable
    to.  If you pass an argument for a task that is not requested, the argument will be silently
    ignored.
    """
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument("--train_file", type=str,
                           help="(1) File with sentences to train on, one per line.")
    argparser.add_argument("--file_to_corrupt", type=str,
                           help="(2) File with sentences to replace words, one per line.")
    argparser.add_argument("--candidates_file", type=str,
                           help="(3) File with candidate sentences to select among, tab-separated.")
    argparser.add_argument("--sentences_file", type=str,
                           help="(3) File with candidate sentences to select among, tab-separated.")
    argparser.add_argument("--word_dim", type=int, default=50,
                           help="(1) Word dimensionality, default=50")
    argparser.add_argument("--max_instances", type=int,
                           help="(1) Maximum number of training examples to use")
    argparser.add_argument("--use_lstm", action='store_true',
                           help="(1) Use LSTM instead of simple RNN")
    argparser.add_argument("--num_epochs", type=int, default=20,
                           help="(1) Maximum number of epochs (will stop early), default=20")
    argparser.add_argument("--factor_base", type=int, default=2,
                           help="(1,2,3) Base of factored indices, default=2")
    argparser.add_argument("--no_tokenize", action='store_true',
                           help="(1,2,3) Do not tokenize input")
    argparser.add_argument("--model_serialization_prefix", default="lexsub",
                           help="(1,2,3) Prefix for saving and loading model files")
    argparser.add_argument("--search_space_size", type=int, default=5000,
                           help="(2) Number of most frequent words to search over as replacement candidates")
    argparser.add_argument("--create_sentence_indices", action="store_true",
                           help="(2,3) If true, output will be [sentence id][tab][sentence]")
    argparser.add_argument("--max_output_sentences", type=int,
                           help="(2,3) If set, limit output to this many corrupted sentences")
    argparser.add_argument("--output_file",
                           help="(2,3) Place to save requested output file")
    argparser.add_argument("--keep_top_k", type=int, default=1,
                           help="(3) Select the top k candidates out of the given alternatives, per line")
    args = argparser.parse_args()

    # Either train or load a model.
    if args.train_file is not None:
        word_replacer = train(args.train_file,
                              args.max_instances,
                              args.factor_base,
                              args.word_dim,
                              args.num_epochs,
                              not args.no_tokenize,
                              args.use_lstm,
                              args.model_serialization_prefix)
    else:
        logger.info("Loading saved model")
        word_replacer = WordReplacer()
        word_replacer.load_model(args.model_serialization_prefix)

    # Sentence corruption, if we were asked to do that.
    if args.file_to_corrupt is not None:
        if args.output_file is None:
            logger.info("Need to specify where to save output with --output_file")
            sys.exit(-1)
        corrupt_sentences(word_replacer,
                          args.file_to_corrupt,
                          args.search_space_size,
                          not args.no_tokenize,
                          args.create_sentence_indices,
                          args.output_file)

    # Selecting most likely candidates, if we were asked to do that.
    if args.candidates_file is not None:
        if args.sentences_file is None:
            select_mostly_likely_candidates(word_replacer,
                                            args.candidates_file,
                                            args.keep_top_k,
                                            not args.no_tokenize,
                                            args.create_sentence_indices,
                                            args.max_output_sentences,
                                            args.output_file)
        else:
            generate_multiple_choice_questions(word_replacer,
                                               args.sentences_file,
                                               args.candidates_file,
                                               not args.no_tokenize,
                                               3,
                                               args.create_sentence_indices,
                                               args.max_output_sentences,
                                               args.output_file)




if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)
    main()
