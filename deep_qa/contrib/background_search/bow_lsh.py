'''
The retrieval method encodes both the background and the query sentences by averaging their word representations,
optionally weighted by their IDF values and does retrieval using a Locality Sensitive Hash (LSH).
We use ScikitLearn's LSH.
'''
import sys
import os
import argparse
import gzip
import pickle
from collections import defaultdict
from typing import List
import numpy

import spacy
from sklearn.neighbors import LSHForest


class BowLsh:
    def __init__(self, serialization_prefix='lsh', use_idf=True):
        self.embeddings = {}
        self.lsh = None
        self.embedding_dim = None
        self.serialization_prefix = serialization_prefix
        # We'll keep track of the min and max vector values so that we can sample values from this
        # interval for UNK if needed.
        self.vector_max = -float("inf")
        self.vector_min = float("inf")
        self.en_nlp = spacy.load('en')
        self.indexed_background = {}  # index -> tokenized background sentence
        self.use_idf = use_idf
        if self.use_idf:
            self.idf_values = {}  # word -> idf_value

    def read_embeddings_file(self, embeddings_file: str):
        with gzip.open(embeddings_file, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.decode('utf-8').strip().split(' ')
                self.embedding_dim = len(fields) - 1
                word = fields[0]
                vector = numpy.asarray(fields[1:], dtype='float32')
                vector_min = min(vector)
                vector_max = max(vector)
                if vector_min < self.vector_min:
                    self.vector_min = vector_min
                if vector_max > self.vector_max:
                    self.vector_max = vector_max
                self.embeddings[word] = vector

    def load_model(self):
        pickled_embeddings_file = open("%s/embeddings.pkl" % self.serialization_prefix, 'rb')
        pickled_lsh_file = open("%s/lsh.pkl" % self.serialization_prefix, 'rb')
        indexed_background_file = open("%s/background.pkl" % self.serialization_prefix, "rb")
        idf_file = open("%s/idf.pkl" % self.serialization_prefix, "rb")
        self.embeddings = pickle.load(pickled_embeddings_file)
        for vector in self.embeddings.values():
            if self.embedding_dim is None:
                self.embedding_dim = len(vector)
                vector_min = min(vector)
                vector_max = max(vector)
                if vector_min < self.vector_min:
                    self.vector_min = vector_min
                if vector_max > self.vector_max:
                    self.vector_max = vector_max
        self.lsh = pickle.load(pickled_lsh_file)
        self.indexed_background = pickle.load(indexed_background_file)
        if self.use_idf:
            self.idf_values = pickle.load(idf_file)

    def save_model(self):
        '''
        We serialize the embedding, indexed background and the LSH here. Note that we delete the embedding and the
        indexed background before serializing the LSH to save memory, and thus those members are unusable after
        calling this method.
        '''
        if not os.path.exists(self.serialization_prefix):
            os.makedirs(self.serialization_prefix)
        pickled_embeddings_file = open("%s/embeddings.pkl" % self.serialization_prefix, "wb")
        pickled_lsh_file = open("%s/lsh.pkl" % self.serialization_prefix, "wb")
        indexed_background_file = open("%s/background.pkl" % self.serialization_prefix, "wb")
        print("\tDumping embeddings", file=sys.stderr)
        pickle.dump(self.embeddings, pickled_embeddings_file)
        print("\tDumping sentences", file=sys.stderr)
        pickle.dump(self.indexed_background, indexed_background_file)
        pickled_embeddings_file.close()
        indexed_background_file.close()
        # Serializing the LSH is memory intensive. Deleting the other members first.
        del self.embeddings
        del self.indexed_background
        if self.use_idf:
            idf_file = open("%s/idf.pkl" % self.serialization_prefix, "wb")
            print("\tDumping IDF", file=sys.stderr)
            pickle.dump(self.idf_values, idf_file)
            idf_file.close()
            del self.idf_values
        print("\tDumping LSH", file=sys.stderr)
        pickle.dump(self.lsh, pickled_lsh_file)
        pickled_lsh_file.close()

    def get_word_vector(self, word, random_for_unk=False):
        if word in self.embeddings:
            vector = self.embeddings[word]
        else:
            # If this is for the background data, we'd want to make new vectors (uniformly sampling
            # from the range (vector_min, vector_max)). If this is for the queries, we'll return a zero vector
            # for UNK because this word doesn't exist in the background data either.
            if random_for_unk:
                vector = numpy.random.uniform(low=self.vector_min, high=self.vector_max,
                                              size=(self.embedding_dim,))
                self.embeddings[word] = vector
            else:
                vector = numpy.zeros((self.embedding_dim,))

        if self.use_idf:
            idf_value = self.idf_values[word] if word in self.idf_values else self.idf_values['@UNK@']
            vector = vector * idf_value

        return vector

    def encode_sentence(self, words: List[str], for_background=False):
        return numpy.mean(numpy.asarray([self.get_word_vector(word, for_background) for word in words]), axis=0)

    def read_background(self, background_file):
        # Read background file, tokenize it and add to indexed_background.
        # Additionally, also compute IDF values of all words in the vocabulary if needed.
        index = 0
        word_doc_frequencies = defaultdict(lambda: 0)
        for sentence in gzip.open(background_file, mode="r"):
            sentence = sentence.decode('utf-8').strip()
            if sentence != '':
                words = [str(w.lower_) for w in self.en_nlp.tokenizer(sentence)]
                if self.use_idf:
                    for word in set(words):
                        word_doc_frequencies[word] += 1
                self.indexed_background[index] = words
                index += 1
        print("Computing IDF values", file=sys.stderr)
        num_docs = len(self.indexed_background)
        for word in word_doc_frequencies:
            self.idf_values[word] = numpy.log(num_docs/word_doc_frequencies[word])
        self.idf_values['@UNK@'] = numpy.log(num_docs)  # Assuming OOV occurred in exactly 1 sentence

    def fit_lsh(self):
        self.lsh = LSHForest(random_state=12345)
        train_data = [self.encode_sentence(self.indexed_background[i],
                                           True) for i in range(len(self.indexed_background))]
        self.lsh.fit(train_data)

    def print_neighbors(self, sentences_file, outfile, num_neighbors=50, sentence_queries=False, num_options=4):
        sentences = []
        indices = []
        for line in open(sentences_file):
            parts = line.strip().split('\t')
            indices.append(parts[0])
            sentences.append(parts[1:])
        test_data = []
        if sentence_queries:
            # Each tuple in sentences is (sentence, label)
            for sentence_parts in sentences:
                words = [str(w.lower_) for w in self.en_nlp.tokenizer(sentence_parts[0])]
                test_data.append(self.encode_sentence(words))
            _, all_neighbor_indices = self.lsh.kneighbors(test_data, n_neighbors=num_neighbors)
        else:
            # Each tuple in sentences is (sentence, options, label)
            for sentence, options_string, _ in sentences:
                options = options_string.split("###")
                if len(options) < num_options:
                    options = options + [''] * (num_options - len(options))
                options = options[:num_options]
                words = [str(w.lower_) for w in self.en_nlp.tokenizer(sentence)]
                test_data.append(self.encode_sentence(words))
                for option in options:
                    option_tokens = [str(w.lower_) for w in self.en_nlp.tokenizer(option)]
                    test_data.append(self.encode_sentence(words + option_tokens))
            num_queries_per_question = 1 + num_options
            similarity_scores, query_neighbor_indices = self.lsh.kneighbors(test_data,
                                                                            n_neighbors=num_neighbors)
            # We need to do some post-processing here to sort the results from all the queries.
            # Note that we are comparing the similarity scores from different queries here. This is not a correct
            # comparison, but we just want to push the not-so-relevant results towards the end so that they can
            # later be pruned if needed.
            num_questions = len(sentences)
            actual_num_neighbors = num_queries_per_question * num_neighbors
            similarity_scores_per_question = numpy.reshape(similarity_scores,
                                                           (num_questions, actual_num_neighbors))
            all_query_neighbor_indices = numpy.reshape(query_neighbor_indices,
                                                       (num_questions, actual_num_neighbors))
            all_neighbor_indices = []
            for neighbor_indices, similarity_scores in zip(all_query_neighbor_indices,
                                                           similarity_scores_per_question):
                # Sorting by similarity scores
                neighbor_indices = [t[1] for t in sorted(zip(similarity_scores, neighbor_indices))]
                filtered_neighbor_indices = []
                seen_neighbors = set([])
                for index in neighbor_indices:
                    if len(filtered_neighbor_indices) >= num_neighbors:
                        break
                    if index not in seen_neighbors:
                        filtered_neighbor_indices.append(index)
                        seen_neighbors.add(index)
                all_neighbor_indices.append(filtered_neighbor_indices)
        with open(outfile, "w") as outfile:
            for i, sentence_neighbor_indices in zip(indices, all_neighbor_indices):
                print("%s\t%s" % (i, "\t".join([" ".join(self.indexed_background[j])
                                                for j in sentence_neighbor_indices])), file=outfile)
            outfile.close()


def main():
    argparser = argparse.ArgumentParser(description="Build a Locality Sensitive Hash and use it for retrieval.")
    argparser.add_argument("--embeddings_file", type=str, help="Gzipped file containing pretrained embeddings \
                           (required for fitting)")
    argparser.add_argument("--background_corpus", type=str, help="Gzipped sentences file (required for fitting)")
    argparser.add_argument("--questions_file", type=str, help="TSV file with indices in the first column \
                           and question in the second (required for retrieval)")
    argparser.add_argument("--retrieved_output", type=str, help="Location where retrieved sentences will be \
                           written (required for retrieval)")
    argparser.add_argument("--serialization_prefix", type=str, help="Loacation where the lsh will be serialized \
                           (default: lsh/)", default="lsh")
    argparser.add_argument("--sentence_queries", help="If this flag is given, queries will be treated\
                           as sentences. If not, they will be treated as question-answer pairs, and the\
                           LSH will get one query per question, and one each per answer option.",
                           action='store_true')
    argparser.add_argument("--no_idf", help="Do not use IDF to weight words while encoding sentences",
                           action='store_true')
    argparser.add_argument("--num_neighbors", type=int, help="Number of background sentences to retrieve",
                           default=50)
    argparser.add_argument("--num_options", type=int, help="Number of options for multiple choice questions",
                           default=4)
    args = argparser.parse_args()
    also_train = False
    use_idf = not args.no_idf
    bow_lsh = BowLsh(args.serialization_prefix, use_idf)
    if args.embeddings_file is not None and args.background_corpus is not None:
        print("Attempting to fit LSH", file=sys.stderr)
        also_train = True
        print("Reading embeddings", file=sys.stderr)
        bow_lsh.read_embeddings_file(args.embeddings_file)
        print("Reading background", file=sys.stderr)
        bow_lsh.read_background(args.background_corpus)
        print("Fitting LSH", file=sys.stderr)
        bow_lsh.fit_lsh()
    if args.questions_file is not None and args.retrieved_output is not None:
        print("Attempting to retrieve", file=sys.stderr)
        if not also_train:
            print("Attempting to load fitted LSH", file=sys.stderr)
            bow_lsh.load_model()
        bow_lsh.print_neighbors(args.questions_file, args.retrieved_output, args.num_neighbors,
                                args.sentence_queries, args.num_options)
    if also_train:
        # We do this after retrieval (if needed) because some members of the class are deleted before LSH
        # is serialized to save memory.
        print("Saving model", file=sys.stderr)
        bow_lsh.save_model()

if __name__ == '__main__':
    main()
