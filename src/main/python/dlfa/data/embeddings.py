import codecs
import gzip
import logging

from typing import Dict, List  # pylint: disable=unused-import

import numpy
from keras.layers import Embedding

from .data_indexer import DataIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class PretrainedEmbeddings:
    @staticmethod
    def initialize_random_matrix(shape, seed=1337):
        numpy_rng = numpy.random.RandomState(seed)  # pylint: disable=no-member
        return numpy_rng.uniform(size=shape, low=0.05, high=-0.05)

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
        words_to_keep = set(data_indexer.words_in_index())
        vocab_size = data_indexer.get_vocab_size()
        embeddings = {}
        embedding_size = None

        # TODO(matt): make this a parameter
        embedding_misses_filename = 'embedding_misses.txt'

        # First we read the embeddings from the file, only keeping vectors for the words we need.
        logger.info("Reading embeddings from file")
        with gzip.open(embeddings_filename, 'rb') as embeddings_file:
            for line in embeddings_file:
                fields = line.decode('utf-8').strip().split(' ')
                if embedding_size is None:
                    embedding_size = len(fields) - 1
                else:
                    if len(fields) - 1 != embedding_size:
                        continue
                word = fields[0]
                if word in words_to_keep:
                    vector = numpy.asarray(fields[1:], dtype='float32')
                    embeddings[word] = vector


        # Now we initialize the weight matrix for an embedding layer, starting with random vectors,
        # then filling in the word vectors we just read.
        logger.info("Initializing pre-trained embedding layer; logging embedding misses to %s",
                    embedding_misses_filename)
        embedding_matrix = PretrainedEmbeddings.initialize_random_matrix((vocab_size, embedding_size))

        # The 2 here is because we know too much about the DataIndexer.  Index 0 is the padding
        # index, and the vector for that dimension is going to be 0.  Index 1 is the OOV token, and
        # we can't really set a vector for the OOV token.
        embedding_misses_file = codecs.open(embedding_misses_filename, 'w', 'utf-8')
        for i in range(2, vocab_size - 1):
            word = data_indexer.get_word_from_index(i)

            # If we don't have a pre-trained vector for this word, we'll just leave this row alone,
            # so the word has a random initialization.
            if word in embeddings:
                embedding_matrix[i] = embeddings[word]
            else:
                print(word, file=embedding_misses_file)
        embedding_misses_file.close()

        # The weight matrix is initialized, so we construct and return the actual Embedding layer.
        return Embedding(input_dim=vocab_size,
                         output_dim=embedding_size,
                         mask_zero=True,
                         weights=[embedding_matrix],
                         trainable=trainable,
                         name="embedding")
