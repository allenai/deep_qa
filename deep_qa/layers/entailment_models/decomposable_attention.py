from keras import backend as K
from keras import activations
from overrides import overrides

from .word_alignment import WordAlignmentEntailment
from ...tensors.backend import switch, apply_feed_forward


class DecomposableAttentionEntailment(WordAlignmentEntailment):
    """
    This layer is a reimplementation of the entailment algorithm described in
    "A Decomposable Attention Model for Natural Language Inference", Parikh et
    al., 2016. The algorithm has three main steps:

    (1) Attend: Compute dot products between all pairs of projections of words
        in the hypothesis and the premise, normalize those dot products to use
        them to align each word in premise to a phrase in the hypothesis and
        vice-versa. These alignments are then used to summarize the aligned
        phrase in the other sentence as a weighted sum. The initial word
        projections are computed using a feed forward NN, F.

    (2) Compare: Pass a concatenation of each word in the premise and the
        summary of its aligned phrase in the hypothesis through a feed forward
        NN, G, to get a projected comparison. Do the same with the hypothesis
        and the aligned phrase from the premise.

    (3) Aggregate: Sum over the comparisons to get a single vector each for
        premise-hypothesis comparison, and hypothesis-premise comparison. Pass
        them through a third feed forward NN (H), to get the entailment
        decision.

    This layer can take either a tuple (premise, hypothesis) or a concatenation
    of them as input.

    Input:

    - Tuple input: a premise sentence and a hypothesis sentence, both with shape ``(batch_size,
      sentence_length, embed_dim)`` and masks of shape ``(batch_size, sentence_length)``
    - Single input: a single tensor of shape ``(batch_size, sentence_length * 2, embed_dim)``, with
      a mask of shape ``(batch_size, sentence_length * 2)``, which we will split in half to get the
      premise and hypothesis sentences.

    Output:

    - Entailment decisions with the given ``output_dim``.

    Parameters
    ----------
    num_hidden_layers: int, optional (default=1)
        Number of hidden layers in each of the feed forward neural nets described above.
    hidden_layer_width: int, optional (default=50)
        Width of each hidden layer in each of the feed forward neural nets described above.
    hidden_layer_activation: str, optional (default='relu')
        Activation for each hidden layer in each of the feed forward neural nets described above.
    final_activation: str, optional (default='softmax')
        Activation to use for the final output.  Should almost certainly be 'softmax'.
    output_dim: int, optional (default=3)
        Dimensionality of the final output.  If this is the last layer in your model, this needs to
        be the same as the number of labels you have.
    initializer: str, optional (default='uniform')
        Will be passed to ``self.add_weight()`` for each of the weight matrices in the feed forward
        neural nets described above.

    Notes
    -----
    premise_length = hypothesis_length = sentence_length below.
    """
    def __init__(self,
                 num_hidden_layers: int=1,
                 hidden_layer_width: int=50,
                 hidden_layer_activation: str='relu',
                 final_activation: str='softmax',
                 output_dim: int=3,
                 initializer: str='uniform',
                 **kwargs):
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation
        self.output_dim = output_dim
        self.initializer = initializer

        # Weights will be initialized in the build method.
        self.premise_length = None
        self.hypothesis_length = None
        self.attend_weights = []  # weights related to F
        self.compare_weights = []  # weights related to G
        self.aggregate_weights = []  # weights related to H
        self.scorer = None
        super(DecomposableAttentionEntailment, self).__init__(**kwargs)

    @overrides
    def build(self, input_shape):
        '''
        This model has three feed forward NNs (F, G and H in the paper). We assume that all three
        NNs have the same hyper-parameters: num_hidden_layers, hidden_layer_width and
        hidden_layer_activation. That is, F, G and H have the same structure and activations. Their
        actual weights are different, though. H has a separate softmax layer at the end.
        '''
        super(DecomposableAttentionEntailment, self).build(input_shape)
        if isinstance(input_shape, list):
            # input_shape is a list containing the shapes of the two inputs.
            self.premise_length = input_shape[0][1]
            self.hypothesis_length = input_shape[1][1]
            # input_dim below is embedding dim for the model in the paper since they feed embedded
            # input directly into this layer.
            self.input_dim = input_shape[0][-1]
        else:
            # NOTE: This will probably fail silently later on in this code if your premise and
            # hypothesis actually have different lengths.
            self.premise_length = self.hypothesis_length = int(input_shape[1] / 2)
            self.input_dim = input_shape[-1]
        attend_input_dim = self.input_dim
        compare_input_dim = 2 * self.input_dim
        aggregate_input_dim = self.hidden_layer_width * 2
        for i in range(self.num_hidden_layers):
            self.attend_weights.append(self.add_weight((attend_input_dim, self.hidden_layer_width),
                                                       name='%s_attend_%d' % (self.name, i),
                                                       initializer=self.initializer))
            self.compare_weights.append(self.add_weight((compare_input_dim, self.hidden_layer_width),
                                                        name='%s_compare_%d' % (self.name, i),
                                                        initializer=self.initializer))
            self.aggregate_weights.append(self.add_weight((aggregate_input_dim, self.hidden_layer_width),
                                                          name='%s_aggregate_%d' % (self.name, i),
                                                          initializer=self.initializer))
            attend_input_dim = self.hidden_layer_width
            compare_input_dim = self.hidden_layer_width
            aggregate_input_dim = self.hidden_layer_width
        self.scorer = self.add_weight((self.hidden_layer_width, self.output_dim),
                                      initializer=self.initializer,
                                      name='%s_score' % self.name)

    @overrides
    def compute_output_shape(self, input_shape):
        # (batch_size, 2)
        if isinstance(input_shape, list):
            return (input_shape[0][0], self.output_dim)
        else:
            return (input_shape[0], self.output_dim)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return None

    @overrides
    def call(self, inputs, mask=None):
        # premise_length = hypothesis_length in the following lines, but the names are kept separate to keep
        # track of the axes being normalized.
        # The inputs can be a two different tensors, or a concatenation. Hence, the conditional below.
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            premise_embedding, hypothesis_embedding = inputs
            # (batch_size, premise_length), (batch_size, hypothesis_length)
            premise_mask, hypothesis_mask = mask
        else:
            premise_embedding = inputs[:, :self.premise_length, :]
            hypothesis_embedding = inputs[:, self.premise_length:, :]
            # (batch_size, premise_length), (batch_size, hypothesis_length)
            premise_mask = None if mask is None else mask[:, :self.premise_length]
            hypothesis_mask = None if mask is None else mask[:, self.premise_length:]
        if premise_mask is not None:
            premise_embedding = switch(K.expand_dims(premise_mask), premise_embedding,
                                       K.zeros_like(premise_embedding))
        if hypothesis_mask is not None:
            hypothesis_embedding = switch(K.expand_dims(hypothesis_mask), hypothesis_embedding,
                                          K.zeros_like(hypothesis_embedding))
        activation = activations.get(self.hidden_layer_activation)
        # (batch_size, premise_length, hidden_dim)
        projected_premise = apply_feed_forward(premise_embedding, self.attend_weights, activation)
        # (batch_size, hypothesis_length, hidden_dim)
        projected_hypothesis = apply_feed_forward(hypothesis_embedding, self.attend_weights, activation)

        ## Step 1: Attend
        p2h_alignment = self._align(projected_premise, projected_hypothesis, premise_mask, hypothesis_mask)
        # beta in the paper (equation 2)
        # (batch_size, premise_length, emb_dim)
        p2h_attention = self._attend(hypothesis_embedding, p2h_alignment, self.premise_length)
        h2p_alignment = self._align(projected_hypothesis, projected_premise, hypothesis_mask, premise_mask)
        # alpha in the paper (equation 2)
        # (batch_size, hyp_length, emb_dim)
        h2p_attention = self._attend(premise_embedding, h2p_alignment, self.hypothesis_length)

        ## Step 2: Compare
        # Equation 3 in the paper.
        compared_premise = self._compare(premise_embedding, p2h_attention)
        compared_hypothesis = self._compare(hypothesis_embedding, h2p_attention)

        ## Step 3: Aggregate
        # Equations 4 and 5.
        # (batch_size, hidden_dim * 2)
        aggregated_input = K.concatenate([K.sum(compared_premise, axis=1), K.sum(compared_hypothesis, axis=1)])
        # (batch_size, hidden_dim)
        input_to_scorer = apply_feed_forward(aggregated_input, self.aggregate_weights, activation)
        # (batch_size, 2)
        final_activation = activations.get(self.final_activation)
        scores = final_activation(K.dot(input_to_scorer, self.scorer))
        return scores

    def _attend(self, target_embedding, s2t_alignment, source_length):
        '''
        Takes target embedding, and source-target alignment attention and produces a weighted average of the
        target embedding per each source word.

        target_embedding: (batch_size, target_length, embed_dim)
        s2t_alignment: (batch_size, source_length, target_length)
        '''
        # We have to explicitly tile tensors below because TF does not broadcast values while performing *.
        # (batch_size, source_length, target_length, embed_dim)
        tiled_s2t_alignment = K.dot(K.expand_dims(s2t_alignment), K.ones((1, self.input_dim)))
        # (batch_size, source_length, target_length, embed_dim)
        tiled_target_embedding = K.permute_dimensions(K.dot(K.expand_dims(target_embedding),
                                                            K.ones((1, source_length))), (0, 3, 1, 2))

        # alpha or beta in the paper depending on whether the source is the premise or hypothesis.
        # sum((batch_size, src_length, target_length, embed_dim), axis=2) = (batch_size, src_length, emb_dim)
        s2t_attention = K.sum(tiled_s2t_alignment * tiled_target_embedding, axis=2)
        return s2t_attention

    def _compare(self, source_embedding, s2t_attention):
        '''
        Takes word embeddings from a sentence, and aggregated representations of words aligned to each of those
        words from another sentence, and returns a projection of their concatenation.

        source_embedding: (batch_size, source_length, embed_dim)
        s2t_attention: (batch_size, source_length, embed_dim)
        '''
        activation = activations.get(self.hidden_layer_activation)
        comparison_input = K.concatenate([source_embedding, s2t_attention])
        # Equation 3 in the paper.
        compared_representation = apply_feed_forward(comparison_input, self.compare_weights, activation)
        return compared_representation

    @overrides
    def get_config(self):
        config = {
                'num_hidden_layers': self.num_hidden_layers,
                'hidden_layer_width': self.hidden_layer_width,
                'hidden_layer_activation': self.hidden_layer_activation,
                'final_activation': self.final_activation,
                'output_dim': self.output_dim,
                'initializer': self.initializer,
                }
        base_config = super(DecomposableAttentionEntailment, self).get_config()
        config.update(base_config)
        return config
