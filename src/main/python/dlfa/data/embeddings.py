import gzip

from typing import Dict, List  # pylint: disable=unused-import

import numpy
from keras.layers import Embedding

from .index_data import DataIndexer

class PretrainedEmbeddings(object):
    @staticmethod
    def get_embedding_layer(embeddings_filename: str, data_indexer: DataIndexer, trainable=False):
        """
        Reads a pre-trained embedding file and generates a Keras Embedding layer that has weights
        initialized to the pre-trained embeddings.  The Embedding layer can either be trainable or
        not.

        We use the DataIndexer to map from the word strings in the embeddings file to the indices
        that we need, and to know which words from the embeddings file we can safely ignore.  If we
        come across a word in DataIndexer that does not show up with the embeddings file, we give
        it a zero vector.

        The embeddings file is assumed to be gzipped, formatted as [word] [dim 1] [dim 2] ...
        """
        words_to_keep = data_indexer.words_in_index()
        vocab_size = data_indexer.get_vocab_size()
        embeddings = {}
        embedding_size = None

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        print("Reading embeddings from file")
        with gzip.open(embeddings_filename, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.strip().split()
                if embedding_size is None:
                    embedding_size = len(fields) - 1
                word = fields[0]
                if word in words_to_keep:
                    vector = numpy.asarray(fields[1:], dtype='float32')
                    embeddings[word] = vector


        # Now we initialize the weight matrix for an embedding layer.
        print("Initializing pre-trained embedding layer")
        embedding_matrix = numpy.zeros((vocab_size, embedding_size))
        empty_embedding = numpy.zeros(embedding_size)

        # The 2 here is because we know too much about the DataIndexer.  Index 0 is the padding
        # index, and the vector for that dimension is going to be 0.  Index 1 is the OOV token, and
        # we can't really set a vector for the OOV token.
        for i in range(2, vocab_size - 1):
            word = data_indexer.get_word_from_index(i)

            # If we don't have a pre-trained vector for this word, return an empty embedding.  Not
            # sure what else we could do that would be better...
            vector = embeddings.get(word, empty_embedding)
            embedding_matrix[i] = vector

        # Now return the embedding layer.
        return Embedding(input_dim=vocab_size,
                         output_dim=embedding_size,
                         mask_zero=True,
                         weights=[embedding_matrix],
                         trainable=trainable)
