from itertools import product
import argparse
from typing import List

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import WeightRegularizer

from nltk import pos_tag
from nltk import FreqDist
from nltk.corpus import reuters
from nltk.corpus import stopwords

from loaders import load_questions, get_pos_set, write_sentences # pylint: disable=import-error
from loaders import  load_parses, read_barrons, get_science_terms # pylint: disable=import-error



# constrained legal span lengths. Upper diagonal band in begin vs. end matrix.
def legal_spans(num_sentence_words, max_span_length):
    for span in product(range(0, num_sentence_words), range(0, num_sentence_words)):
        if span[0] >= span[1] or span[1] - span[0] > max_span_length:
            continue
        yield span


def create_training_examples(statement_list: List[List[str]], trees, training_spans,
                             training: bool, max_span_length, num_sentence_words,
                             args, pos_tags, constituents, k=1):
    """
    This function defines a training set for span prediction.
    I computes features (inputs) for eveyspans.
    Multiple spans are extracted for a given statement, and features are computed
    for each.
    :input statement_list: list of token lists.
    :input trees: parse tree list, aligned with statement_list
    :training: bool, indicates whether annotation (True/False) is computed.
    :k: int, number of negative span examples per positive. Default: 1 (1:1 ratio)
    """
    print('building features...')

    word_frequencies = FreqDist([w.lower() for w in reuters.words()])
    # list of science tokens & science multiword expressions
    science_tokens = get_science_terms(args.data_path)
    science_expressions = get_science_terms(args.data_path, False)
    stop_words = set(stopwords.words('english'))



    true_examples, false_examples, examples_per_sentence = [], [], []
    span_indexes = []

    # loop over all statements
    for i, statement in enumerate(statement_list):
        print(i)

        span_index = [] #enumerating all spans for this sentence.
        pos_tags_this_statement = pos_tag(statement)
        sentence_word_frequencies = [word_frequencies.freq(token.lower()) \
                        for token in statement]
        tree = trees[i]
        false_examples_this_instance = []
        sentence_candidate_span_examples = []   # when training.

        # loop across different spans for given sentence
        for span in legal_spans(num_sentence_words, max_span_length):  #globally legal
            if span[1] > len(statement):
                continue
            if span[0] > len(statement):
                break
            span_index.append(span)

            # extract scalar features for this span. [position will not be used]
            f_bias = 1
            f_length = span[1] - span[0]
            #f_begin = span[0]
            #f_end = span[1]
            #f_dist_to_end0 = len(statement) - span[0]
            #f_dist_to_end1 = len(statement) - span[1]

            # list of tokens of this span
            span_tokens = statement[span[0]: span[1]]

            # feature: span contains at least one science token
            f_science_token = bool( \
                set(span_tokens).intersection(science_expressions))



            f_science_token_count = 0   # counting # of science tokens in span
            max_token_length = 0        # in this span.
            for token in span_tokens:
                f_science_token_count += int(token in science_tokens)
                max_token_length = max(max_token_length, len(token))

            f_max_token_length = np.log(max_token_length)

            # feature: relative word frequency average
            # with numerical stability/ avoiding -inf
            f_avg_word_frequency = 1e-10+np.mean(sentence_word_frequencies[span[0]: span[1]])
            f_avg_word_frequency = np.log(f_avg_word_frequency)

            # feature: begin with stop word?
            f_stop_word_begin = bool(span_tokens[0] in stop_words)

            # POS indicator (one-hot)
            f_pos = np.zeros([len(pos_tags)])

            # Bag-of-POS-tags for this span.
            for token, tag in pos_tags_this_statement[span[0]:span[1]]:
                f_pos[pos_tags.index(tag)] += 1.0

            # feature: POS indicator for span beginning
            f_pos_beginning = np.zeros([len(pos_tags)])
            f_pos_beginning[pos_tags.index(pos_tags_this_statement[span[0]][1])] = 1.0

            # feature: POS indicator for span end
            f_pos_end = np.zeros([len(pos_tags)])
            f_pos_end[pos_tags.index(pos_tags_this_statement[span[1]-1][1])] = 1.0

            # feature: POS bigram indicator
            # define extended POS tag set with additional begin and end symbols for bigrams.
            # pos_tags_bigram = pos_tags + ["POS_BEGIN", "POS_END"]

            # for POS bigrams.
            # pos_tags_square = [x for x in product(pos_tags_bigram, pos_tags_bigram)]

            # f_pos_bigram = np.zeros([len(pos_tags_square)])

            # obtaining the POS bigram
            # for position in range(-1, f_length):
                # boundary cases: start of span and end of span.
            #    if position == -1:
            #        tag1 = 'POS_BEGIN'
            #        _, tag2 = pos_tags_this_statement[span[0]]
            #    elif position == f_length -1:
            #        _, tag1 = pos_tags_this_statement[span[0]+position]
            #        tag2 = 'POS_END'
            #    #normal case: inside span.
            #    else:
            #        _, tag1 = pos_tags_this_statement[span[0] + position]
            #        _, tag2 = pos_tags_this_statement[span[0] + position + 1]
            #
            #    f_pos_bigram[pos_tags_square.index( ( tag1, tag2 ) )] += 1.0

            # constituent tree features

            tree_position = tree.treeposition_spanning_leaves(span[0], span[1])

            # smallest subtree in constituent parse, containing this span.
            smallest_subtree = tree[tree_position[:-1]]
            constituent_tag = smallest_subtree.label()

            # feature: is this span a constituent parse subtree span?
            f_span_match = bool(span[1]-span[0] == len(smallest_subtree))

            # constituency parse label indicator
            f_span_constituent = np.zeros([len(constituents)])
            f_span_constituent[constituents.index(constituent_tag)] = 1.0

            # constituency parse label indicator with indication for large spans.
            f_span_constituent_big = np.zeros([len(constituents)])
            f_span_constituent_big[constituents.index(constituent_tag)] = (f_length > 2)


            # leave out position features:
            ####  f_begin, f_end, f_dist_to_end0, f_dist_to_end1,

            #now collect all features:
            f_scalars = np.array([f_bias, f_span_match, f_length,
                                  f_science_token, f_avg_word_frequency,
                                  f_stop_word_begin,
                                  f_max_token_length,
                                  f_science_token_count])

            # these are all features for this span, in a np array.
            feature_vector = np.concatenate((f_scalars, f_pos, f_pos_beginning,
                                             f_pos_end, f_span_constituent,
                                             f_span_constituent_big))



            # provide True/False annotation in case the data is used for training.
            if training:
                if span == training_spans[i]:
                    #positive example
                    true_examples.append(feature_vector)
                    sentence_candidate_span_examples.append((feature_vector, True))
                else:
                    #negative example
                    false_examples_this_instance.append(feature_vector)
                    sentence_candidate_span_examples.append((feature_vector, False))
            else:
                sentence_candidate_span_examples.append(feature_vector)

        span_indexes.append(span_index)
        examples_per_sentence.append(sentence_candidate_span_examples)


        # select at random k negative spans as training examples. default 1:1
        if training:
            for random_index in np.random.randint(0, len(false_examples_this_instance), k):
                false_examples.append(false_examples_this_instance[random_index])

    print(len(true_examples), 'True span examples.')
    print(len(false_examples), 'False span examples.')

    # collect true and false examples [inputs]
    all_examples = np.concatenate((np.asarray(false_examples), np.asarray(true_examples)))

    # collect annotations for each example (True/False target outputs)
    false_span_labels = np.zeros([len(false_examples)])
    true_span_labels = np.ones([len(true_examples)])
    all_labels = np.concatenate((false_span_labels, true_span_labels))

    return all_examples, all_labels, examples_per_sentence, span_indexes




def main():
    argparser = argparse.ArgumentParser(description="Run span prediction model for fill-the-gap questions")
    argparser.add_argument("--data_path", type=str, default='../data/',
                           help="path to Omnibus-Gr04/Omnibus-Gr04/Barron's data."+\
                           "If this doesn't work specify globally in loaders.py")
    argparser.add_argument("--output_file", type=str, default='barrons_predictions-1.txt',
                           help="File with predicted examples, one per line.")
    argparser.add_argument("--barrons_file", type=str,
                           help="Filepath of Barrons-1.sentences.txt")
    argparser.add_argument("--evaluate", type=bool, default=True,
                           help="run per-sentence evaluation")
    argparser.add_argument("--barrons", type=bool, default=True,
                           help="generate examples for Barron's statements.")
    argparser.add_argument("--neural", type=bool, default=False,
                           help="additional intermediate layer")
    argparser.add_argument("--plot", type=bool, default=False,
                           help="whether to create plot with feature importances")
    args = argparser.parse_args()



    # load fill-the-gap data
    training_examples = load_questions(args.data_path)
    training_statements, training_spans = zip(*training_examples)

    #'Barrons-4thGrade.sentences-d1/Barrons-1.sentences.txt'
    barrons_statements = read_barrons(args.barrons_file)

    # getting parses. Assume CoreNLP is installed.
    # Parse tree annotation must be run before, both for training and new data.
    write_sentences(training_statements, filename='training.txt')
    write_sentences(barrons_statements, filename='barrons.txt')

    barron_trees = load_parses(filename='barrons.txt.json')
    training_trees = load_parses(filename='training.txt.json')


    # general characteristics of sentences/ spans
    max_span_length = max([span[1]-span[0] for span in training_spans])     #11
    num_sentence_words = max([len(stmnt) for stmnt in training_statements])    #46




    ### build span dataset: {spans (their feature repr.)} --> {True/False}
    pos_tags = get_pos_set(list(training_statements) + barrons_statements)

    # list of legal constituent labels
    constituents = set()
    for tree in training_trees + barron_trees:
        for subtree in tree.subtrees(lambda t: t.height() > 1):
            constituents.add(subtree.label())
    constituents = sorted(list(constituents))





    #### load training data
    all_examples, all_labels, examples_per_sentence, _ = \
                    create_training_examples(training_statements, training_trees,
                                             training_spans, True,
                                             max_span_length, num_sentence_words,
                                             args, pos_tags, constituents)


    # shuffle training data order
    n_examples, n_features = all_examples.shape
    perm = np.random.permutation(n_examples)
    all_examples = all_examples[perm, :]
    all_labels = all_labels[perm]
    print(n_examples, 'examples; ', n_features, 'features')


    # split into training and (preliminary) validation part
    cut = 180
    x_train = all_examples[:cut, :]
    x_test = all_examples[cut:, :]
    y_train = all_labels[:cut]
    y_test = all_labels[cut:]
    examples_per_sentence_train = examples_per_sentence[:cut]
    examples_per_sentence_test = examples_per_sentence[cut:]


    ##### define keras model
    model = Sequential()
    regul = 0.01

    # extra neural layer.
    if args.neural:
        n_latent = 5
        model.add(Dense(output_dim=n_latent, input_dim=n_features, activation='tanh',
                        W_regularizer=WeightRegularizer(l1=regul)))
        model.add(Dense(output_dim=1, input_dim=n_latent, activation='sigmoid',
                        W_regularizer=WeightRegularizer(l1=regul)))

    # sigmoid model
    if not args.neural:
        model.add(Dense(output_dim=1, input_dim=n_features, activation='sigmoid',
                        W_regularizer=WeightRegularizer(l2=regul)))


    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    # training
    model.fit(x_train, y_train, nb_epoch=200, batch_size=8)


    # per-span evaluation
    loss_and_metrics = model.evaluate(x_train, y_train, batch_size=100)
    loss_and_metrics_test = model.evaluate(x_test, y_test, batch_size=100)
    print('Training performance (sigmoid):', loss_and_metrics)
    print('Validation performance (sigmoid):', loss_and_metrics_test)


    # per-sentence evaluation.
    # evaluate on all legal spans for a given sentence:
    if args.evaluate:
        accuracies = []
        lengths = []
        for sentence_spans in examples_per_sentence_train:
            lengths.append(len(sentence_spans))
            scores_this_sentence = []
            correct_index = -1
            for i, (features, truth) in enumerate(sentence_spans):
                x_input = np.reshape(features, [1, n_features])
                #y_input = np.atleast_1d(truth)
                score = model.predict_proba(x_input, verbose=False)
                scores_this_sentence.append(score)
                if truth:
                    correct_index = i
                #loss, accuracy = model.evaluate(x_input, y_input, batch_size=1)
                #sentence_accuracies.apopend(accuracy)
            prediction = np.argmax(scores_this_sentence)
            accurate = (prediction == correct_index)
            accuracies.append(accurate)
        print(np.mean(accuracies), 'Training accuracy (sentence level)')
        print('Average legal spans per sentence:', np.mean(lengths))

        # same as above, but for evaluation examples. Too lazy to properly factor out.
        accuracies = []
        lengths = []
        for sentence_spans in examples_per_sentence_test:
            lengths.append(len(sentence_spans))
            scores_this_sentence = []
            correct_index = -1
            for i, (features, truth) in enumerate(sentence_spans):
                x_input = np.reshape(features, [1, n_features])
                #y_input = np.atleast_1d(truth)
                score = model.predict_proba(x_input, verbose=False)
                scores_this_sentence.append(score)
                if truth:
                    correct_index = i
                #loss, accuracy = model.evaluate(x_input, y_input, batch_size=1)
                #sentence_accuracies.apopend(accuracy)
            prediction = np.argmax(scores_this_sentence)
            accurate = (prediction == correct_index)
            accuracies.append(accurate)
        print(np.mean(accuracies), 'Validation accuracy (sentence level)')



    # weight interpretation/ plot
    if args.plot:
        weights = model.layers[0].get_weights()[0]
        #important_features = np.where(np.abs(weights) > 0.05)[0]

        feature_names = ["f_bias",
                         "f_span_match",
                         "f_length",
                         "f_science_token",
                         "f_avg_word_frequency",
                         "f_stop_word_begin",
                         "f_max_token_length",
                         "f_science_token_count",
                        ]
        feature_names += pos_tags
        feature_names += ["begin_"+x for x in pos_tags]
        feature_names += ["end_"+x for x in pos_tags]
        feature_names += constituents
        feature_names += ["big_"+x for x in constituents]

        #order = np.argsort(np.abs(weights))

        plt.stem(weights)
        plt.xticks(range(0, len(feature_names)), feature_names, rotation='vertical')
        plt.grid()
        plt.show()



    # generating predictions for barron's
    if args.barrons:
        barron_statements = [tree.leaves() for tree in barron_trees]
        for statement in barron_statements:
            try:
                statement[statement.index('-LRB-')] = '('
            except ValueError:
                pass
            try:
                statement[statement.index('-RRB-')] = ')'
            except ValueError:
                pass

        # compute features for barron's
        _, _, barron_span_features, span_indexes = \
        create_training_examples(barron_statements, barron_trees, False, False,
                                 max_span_length, num_sentence_words, args, pos_tags,
                                 constituents)


        # identify span for each sentence with highest score
        predicted_spans = []
        for i_sent, sentence_spans in enumerate(barron_span_features):
            scores_this_sentence = []
            for features in sentence_spans:
                x_input = np.reshape(features, [1, n_features])
                #y_input = np.atleast_1d(truth)
                score = model.predict_proba(x_input, verbose=False)
                scores_this_sentence.append(score)
            predicted_span = span_indexes[i_sent][np.argmax(scores_this_sentence)]
            predicted_spans.append(predicted_span)

        # write predictions to file
        with open(args.output_file, 'w') as writefile:
            for i in range(0, len(barron_statements)):
                printstring = barron_statements[i]  # full sentence without gap.
                gap = predicted_spans[i]
                gap_tokens = printstring[gap[0]:gap[1]]
                printstring[gap[0]:gap[1]] = ['_____']#*len(gap_tokens)
                printstring = " ".join(printstring) + '\t'
                printgap = " ".join(gap_tokens) + '\n'
                writefile.write(printstring+printgap)


if __name__ == "__main__":
    main()
