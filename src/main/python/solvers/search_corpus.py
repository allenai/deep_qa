from __future__ import print_function
import sys
import argparse
import codecs
import gzip
import pickle
import numpy
from sklearn.neighbors import LSHForest
from keras.layers import Input
from keras.models import Model

from memory_network import MemoryNetworkSolver
from knowledge_backed_scorers import MemoryLayer


class CorpusSearcher(object):
    """Encodes a corpus of sentences using a sentence encoder (e.g., as LSTM), then allows for
    nearest-neighbor search on the sentence encodings.

    We perform the nearest-neighbor search using a scikit-learn's locality sensitive hash (LSH).
    All variable names involving "lsh" in this class refer to "locality sensitive hash".

    TODO(matt): this should be a subclass of MemoryNetworkSolver, where get_nearest_neighbors
    returns a list in memory, instead of writing to a file.
    """
    def __init__(self, corpus_path):
        """
        corpus_path: path to a gzipped file containing sentences, one sentence per line.
        """
        self.corpus_path = corpus_path
        self.encoder_model = None
        self.nn_solver = None
        self.max_sentence_length = None
        self.lsh = LSHForest(random_state=12345)
        self.sentence_index = {}  # Dict: index -> sentence

    def load_encoder(self, trained_model_name_prefix):
        # Load the memory network solver, and make an encoder model out of it
        # with just the embedding and LSTM layers. But store the original solver
        # as well, since we need it to index words in sentences.
        memory_network_solver = MemoryNetworkSolver('memory')
        memory_network_solver.load_model(trained_model_name_prefix,
                                         custom_objects={"MemoryLayer": MemoryLayer})
        memory_network_model = memory_network_solver.model
        self.nn_solver = memory_network_solver
        embedding_layer = None
        encoder_layer = None
        for layer in memory_network_model.layers:
            if layer.name == "embedding":
                embedding_layer = layer
            elif layer.name == "encoder":
                encoder_layer = layer
        if not embedding_layer or not encoder_layer:
            raise RuntimeError, "Memory network model does not have the required layers"
        # Get the input length that the encoder expects.
        input_length = memory_network_model.get_input_shape_at(0)[0][1]
        self.max_sentence_length = input_length
        input_layer = Input(shape=(input_length,), dtype="int32")
        embedded_input = embedding_layer(input_layer)
        encoded_input = encoder_layer(embedded_input)
        self.encoder_model = Model(input=input_layer, output=encoded_input)
        # Loss and optimizer do not matter since we're not going to train it. But it needs
        # to be compiled to use it for prediction.
        self.encoder_model.compile(loss="mse", optimizer="adam")

    def initialize_lsh(self, batch_size=100):
        """
        This method encodes the corpus in batches, using encoder_model initialized above.  After
        the whole corpus is encoded, we pass the vectors off to sklearn's LSHForest.fit() method.
        """
        corpus_file = gzip.open(self.corpus_path)
        def _get_generator():
            while True:
                sentences = [corpus_file.readline().decode("utf-8") for _ in range(batch_size)]
                if not sentences[-1]:
                    # Readline keeps returning '' (no newline at the end) after the end of file is reached.
                    break
                yield sentences
        generator = _get_generator()
        encoded_sentences = []
        for lines in generator:
            indexed_lines = [(i, line.strip()) for i, line in enumerate(lines)]

            # index_inputs returns a dictionary mapping sentence indices to lists of sentences.  In
            # this case, each list will have a single sentence.
            mapped_indices = self.nn_solver.index_inputs(indexed_lines, for_train=False)
            assert len(mapped_indices) == len(indexed_lines)
            # Indices of the current batch of sentences from the generator.
            corpus_indices = []
            for index, line in indexed_lines:
                # Use index to map sentences to word indices. The index here is reused for
                # each batch yielded by the generator.
                if index in mapped_indices:
                    # Mapped indices for each index is a list containing indices of only one
                    # sentence.
                    corpus_indices.append(mapped_indices[index][0])
                    self.sentence_index[len(self.sentence_index)] = line
            # TODO(pradeep): Try not to use the function of a member object. May be expose
            # the padding function in nn_solver.
            padded_corpus_indices = self.nn_solver.data_indexer.pad_indices(
                    corpus_indices, self.max_sentence_length)
            encoder_input = numpy.asarray(padded_corpus_indices, dtype='int32')
            current_batch_encoded_sentences = self.encoder_model.predict(encoder_input)
            for encoded_sentence in current_batch_encoded_sentences:
                encoded_sentences.append(encoded_sentence)
        encoded_sentences = numpy.asarray(encoded_sentences)
        self.lsh.fit(encoded_sentences)

    def save_lsh(self, serialization_prefix):
        lsh_file = open("%s_lsh.pkl" % serialization_prefix, "wb")
        sentence_index_file = open("%s_index.pkl" % serialization_prefix, "wb")
        pickle.dump(self.lsh, lsh_file)
        pickle.dump(self.sentence_index, sentence_index_file)
        lsh_file.close()
        sentence_index_file.close()

    def load_lsh(self, serialization_prefix):
        lsh_file = open("%s_lsh.pkl" % serialization_prefix, "rb")
        sentence_index_file = open("%s_index.pkl" % serialization_prefix, "rb")
        self.lsh = pickle.load(lsh_file)
        self.sentence_index = pickle.load(sentence_index_file)
        lsh_file.close()
        sentence_index_file.close()

    def get_nearest_neighbors(self, input_sentences, outfile_name, num_neighbors=10):
        '''
        input_sentences: list((index, sentence))
        outfile_name: str: Closest sentences will be printed here.
        num_neighbors (int)
        '''
        mapped_indices = self.nn_solver.index_inputs(input_sentences, for_train=False)
        input_sentence_ids = []
        input_sentence_indices = []
        for sentence_id, sentence_indices in mapped_indices.items():
            input_sentence_ids.append(sentence_id)
            input_sentence_indices.append(sentence_indices[0])
        # Pad indices
        # TODO(pradeep): Try not to use the function of a member object. May be expose
        # the padding function in nn_solver.
        input_sentence_indices = self.nn_solver.data_indexer.pad_indices(
                input_sentence_indices, self.max_sentence_length)
        encoded_input_sentences = self.encoder_model.predict(numpy.asarray(input_sentence_indices))
        _, all_nearest_neighbor_indices = self.lsh.kneighbors(
                encoded_input_sentences, n_neighbors=num_neighbors)
        outfile = codecs.open(outfile_name, "w", "utf-8")
        for i, nn_indices in enumerate(all_nearest_neighbor_indices):
            outstring = "%s\t%s" % (input_sentence_ids[i],
                                    "\t".join([self.sentence_index[nn_index] for nn_index in nn_indices]))
            print(outstring, file=outfile)
        outfile.close()

def main():
    argparser = argparse.ArgumentParser(description="Regenerate background data using trained encoder")
    argparser.add_argument('model_serialization_prefix', type=str, help="Path to save/load LSH")
    argparser.add_argument('encoder_prefix', type=str, help="Path to load saved encoder model")
    argparser.add_argument('--corpus_path', type=str, help="Location of corpus to index")
    argparser.add_argument('--query_file', type=str, help="Query file, tsv (index, sentence)")
    argparser.add_argument('--output_file', type=str, default="out.txt")
    args = argparser.parse_args()

    corpus_searcher = CorpusSearcher(args.corpus_path)
    corpus_searcher.load_encoder(args.encoder_prefix)
    if args.corpus_path:
        corpus_searcher.initialize_lsh()
        corpus_searcher.save_lsh(args.model_serialization_prefix)
    else:
        print("Loading saved LSH", file=sys.stderr)
        corpus_searcher.load_lsh(args.model_serialization_prefix)
    if args.query_file:
        test_lines = [line.strip().split("\t") for line in codecs.open(args.query_file, "r", "utf-8")]
        corpus_searcher.get_nearest_neighbors(test_lines, args.output_file)


if __name__ == "__main__":
    main()
