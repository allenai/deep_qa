Similarity functions take a pair of tensors with the same shape, and compute a similarity function
on the vectors in the last dimension.  For example, the tensors might both have shape
`(batch_size, sentence_length, embedding_dim)`, and we will compute some function of the two
vectors of length `embedding_dim` for each position `(batch_size, sentence_length)`, returning a
tensor of shape `(batch_size, sentence_length)`.

The similarity function could be as simple as a dot product, or it could be a more complex,
parameterized function.  The SimilarityFunction class exposes an API for a Layer that wants to
allow for multiple similarity functions, such as for initializing and returning weights.

If you want to compute a similarity between tensors of different sizes, you need to first tile them
in the appropriate dimensions to make them the same before you can use these functions.  The
Attention and MatrixAttention layers do this.
