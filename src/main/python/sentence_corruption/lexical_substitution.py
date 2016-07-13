import sys
import pickle
import codecs
import argparse

from index_data import DataIndexer
from keras.models import Model, model_from_json
from keras.layers import Input, LSTM, Embedding, Dropout, merge, TimeDistributed, Dense
from keras.callbacks import EarlyStopping

class WordReplacer(object):
    def __init__(self):
        self.data_indexer = DataIndexer()
    
    def process_data(self, sentences, max_length=None, factor_base=2):
        #TODO: Deal with OOV
        sentence_lengths, indexed_sentences = self.data_indexer.index_data(sentences, 
                max_length)
        # We want the inputs to be words 0..n-1 and targets to be words 1..n in all 
        # sentences, so that at each time step t, p(w_{t+1} | w_{0}..w_{t}) will be 
        # predicted.
        input_array = indexed_sentences[:,:-1] # Removing the last word from each sentence
        target_array = indexed_sentences[:,1:] # Removing the first word from each sent.
        factored_target_arrays = self.data_indexer.factor_target_indices(target_array, 
                base=factor_base)
        return sentence_lengths, input_array, factored_target_arrays

    def train_model(self, sentences, word_dim=50, factor_base=2, num_epochs=20, 
            model_serialization_prefix="lexsub"):
        _, input_array, factored_target_arrays = self.process_data(sentences, 
                factor_base=factor_base)
        vocab_size = self.data_indexer.get_vocab_size()
        num_factors = len(factored_target_arrays)
        model_input = Input(shape=input_array.shape[1:], dtype='int32')
        embedding = Embedding(input_dim=vocab_size, output_dim=word_dim, mask_zero=True)
        embedded_input = embedding(model_input)
        regularized_embedded_input = Dropout(0.5)(embedded_input)
        # Bidirectional LSTM = Two LSTMs, with the second one processing the input 
        # backwards, and the two outputs concatenated.
        # Return sequences returns output at every timestep, instead of just the last one.
        forward_lstm = LSTM(output_dim=word_dim/2, return_sequences=True, 
                name='forward_lstm')
        backward_lstm = LSTM(output_dim=word_dim/2, go_backwards=True, 
                return_sequences=True, name='backward_lstm')
        forward_lstm_out = forward_lstm(regularized_embedded_input)
        backward_lstm_out = backward_lstm(regularized_embedded_input)
        bidirectional_lstm_out = merge([forward_lstm_out, backward_lstm_out],
                mode='concat')
        regularized_lstm_out=Dropout(0.2)(bidirectional_lstm_out)
        model_outputs = []
        # Make as many output layers as there are factored target arrays, and the same size
        for i in range(num_factors):
            # TimeDistributed(layer) makes layer accept an additional time distribution.
            # i.e. if layer takes a n-dimensional input, TimeDistributed(layer) takes a
            # n+1 dimensional input, where dimension1 is time.
            factor_output = TimeDistributed(Dense(output_dim=factor_base, 
                    activation='softmax', name='factor_output_%d'%i))
            model_outputs.append(factor_output(regularized_lstm_out))
        model = Model(input=model_input, output=model_outputs)
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        print >>sys.stderr, model.summary()
        early_stopping = EarlyStopping()
        model.fit(input_array, factored_target_arrays, nb_epoch=num_epochs, 
                validation_split=0.1, callbacks=[early_stopping])
        data_indexer_pickle_file = open("%s_di.pkl"%model_serialization_prefix, "wb")
        pickle.dump(self.data_indexer, data_indexer_pickle_file)
        model_config = model.to_json()
        model_config_file = open("%s_config.json"%model_serialization_prefix, "w")
        print >>model_config_file, model_config
        model.save_weights("%s_weights.h5"%model_serialization_prefix, overwrite=True)
        self.model = model

    def get_model_input_shape(self):
        return self.model.get_input_shape_at(0)

    def load_model(self, model_serialization_prefix="lexsub"):
        self.data_indexer = pickle.load(open("%s_di.pkl"%model_serialization_prefix))
        self.model = model_from_json(open("%s_config.json"%model_serialization_prefix).read())
        self.model.load_weights("%s_weights.h5"%model_serialization_prefix)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        print >>sys.stderr, self.model.summary()

    def get_substitutes(self, sentences, locations, num_substitutes=5, 
            train_sequence_length=None):
        '''
        sentences (list(str)): List of sentences with words that need to be substituted
        locations (list(int)): List of indices, the same size as sentences, containing
            indices of words in sentences that need to be substituted
        train_sequence_length (int): Length of sequences the model was trained on
        '''
        sentence_lengths, indexed_sentences, _ = self.process_data(sentences, 
                max_length=train_sequence_length+1) # +1 because the last word would be stripped
        all_word_predictions = self.model.predict(indexed_sentences)
        all_substitutes = []
        for sentence_id, (sentence_length, location) in enumerate(zip(sentence_lengths, 
                locations)):
            prediction_length = sentence_length - 1 # Ignore the starting <s> symbol
            # Take all "digits" of the prediction of appropriate lengths corresponding 
            # to this sentence, and then get the corresponding location probabilities
            # to make sure location points to the appropriate word.
            word_predictions = [predictions[sentence_id][-prediction_length:][location] for 
                    predictions in all_word_predictions]
            sorted_substitutes = self.data_indexer.unfactor_probabilities(word_predictions)
            all_substitutes.append(sorted_substitutes[:num_substitutes])
        return all_substitutes
            
if __name__=="__main__":
    argparser = argparse.ArgumentParser(description="Generate lexical substitutes using a \
            bidirectional LSTM")
    argparser.add_argument("--train_file", type=str, help="File with sentences to train on,\
            one per line.")
    argparser.add_argument("--test_file", type=str, help="Tsv file with indices and \
            sentences to replace words, one per line.")
    argparser.add_argument("--word_dim", type=int, help="Word dimensionality, default=50",
            default=50)
    argparser.add_argument("--factor_base", type=int, help="Base of factored indices, \
            default=2", default=2)
    argparser.add_argument("--num_epochs", type=int, help="Maximum number of epochs (will\
            stop early), default=20", default=20)
    args = argparser.parse_args()
    word_replacer = WordReplacer()
    if args.train_file is not None:
        print >>sys.stderr, "Reading training data"
        train_sentences = [x.strip().lower() for x in codecs.open(args.train_file, 
            "r", "utf-8")]
        word_replacer.train_model(train_sentences, factor_base=args.factor_base, 
                num_epochs=args.num_epochs)
    else:
        print >>sys.stderr, "Loading saved model"
        word_replacer.load_model()

    if args.test_file is not None:
        print >>sys.stderr, "Reading test data"
        # TODO: The test file is expected to contain indices for replacement. This
        # is because randomly replacing any word may not be helpful. The assumption
        # is that there is an external process which decides what the important words 
        # are. Ideally, the process should be in this module.
        # Note: The indices will be used with sentences tokenized using NLTK's word
        # tokenizer.
        locations, test_sentences = zip(*[x.strip().lower().split("\t") for x in 
            codecs.open(args.test_file, "r", "utf-8")])
        locations = [int(x) for x in locations]
        train_sequence_length = word_replacer.get_model_input_shape()[1]
        substitutes = word_replacer.get_substitutes(test_sentences, locations, 
                train_sequence_length=train_sequence_length)
        # TODO: Better output format.
        outfile = codecs.open("out.txt", "w", "utf-8")
        for logprob_substitute_list in substitutes:
            print >>outfile, "\t".join("%s %f"%(word, logprob) for logprob, word in 
                    logprob_substitute_list)
