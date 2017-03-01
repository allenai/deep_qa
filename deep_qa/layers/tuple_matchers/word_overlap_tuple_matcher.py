from keras import backend as K
from keras import initializations, activations
from keras.layers import Layer
from overrides import overrides

from ...tensors.backend import switch, apply_feed_forward


class WordOverlapTupleMatcher(Layer):
    r"""
    This layer takes as input two tensors corresponding to two tuples, an answer tuple and a background tuple,
    and calculates the degree to which the background tuple `entails` the answer tuple.  Entailment is
    determined by generating a set of entailment features from the tuples (the number of
    entailment_features = number of tuple slots), and then passing these features into a shallow NN to get an
    entailment score.
    Each entailment feature is currently made by comparing the corresponding slots in the two tuples and
    determining the degree of lexical overlap using the formula:
        :math:`normalized overlap_s = \dfrac{|A_s \cap B_s|}{|A_s|}`
    where :math:`s` is the index of the slot, :math:`A_s` is answer tuple slot :math:`s` and :math:`B_s` is
    background tuple slot :math:`s`.

    Inputs:
        - tuple_1_input (the answer tuple), shape ``(batch size, num_slots, num_slot_words_t1)``,
          any mask is ignored.  Here num_slot_words_t1 is the maximum number of words in each of the
          slots in tuple_1.
        - tuple_2_input (the background_tuple), shape ``(batch size, num_slots, num_slot_words_t2)``,
          and again, any corresponding mask is ignored. As above, num_slot_words_t2 is the
          maximum number of words in each of the slots in tuple_2. This need not match tuple 1.

    Output:
        - entailment score, shape ``(batch, 1)``

    Parameters
    ----------
    - num_hidden_layers : int, default=1
        Number of hidden layers in the shallow NN.

    - hidden_layer_width : int, default=4
        The number of nodes in each of the NN hidden layers.

    - initialization : string, default='glorot_uniform'
        The initialization of the NN weights

    - hidden_layer_activation : string, default='relu'
        The activation of the NN hidden layers

    - final_activation : string, default='sigmoid'
        The activation of the NN output layer

    Notes
    -----
    This layer is incompatible with the WordsAndCharacters tokenizer.
    """

    def __init__(self, num_hidden_layers: int=1, hidden_layer_width: int=4,
                 initialization: str='glorot_uniform', hidden_layer_activation: str='tanh',
                 final_activation: str='sigmoid', **kwargs):
        self.supports_masking = True
        # Parameters for the shallow neural network
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_init = initialization
        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation
        self.hidden_layer_weights = []
        self.score_layer = None
        super(WordOverlapTupleMatcher, self).__init__(**kwargs)

    def get_config(self):
        base_config = super(WordOverlapTupleMatcher, self).get_config()
        config = {'num_hidden_layers': self.num_hidden_layers,
                  'hidden_layer_width': self.hidden_layer_width,
                  'initialization': self.hidden_layer_init,
                  'hidden_layer_activation': self.hidden_layer_activation,
                  'final_activation': self.final_activation}
        config.update(base_config)
        return config

    def build(self, input_shape):
        super(WordOverlapTupleMatcher, self).build(input_shape)

        # Add the weights for the hidden layers.
        hidden_layer_input_dim = input_shape[0][1]
        for i in range(self.num_hidden_layers):
            hidden_layer = self.add_weight(shape=(hidden_layer_input_dim, self.hidden_layer_width),
                                           initializer=initializations.get(self.hidden_layer_init),
                                           name='%s_hiddenlayer_%d' % (self.name, i))
            self.hidden_layer_weights.append(hidden_layer)
            hidden_layer_input_dim = self.hidden_layer_width
        # Add the weights for the final layer.
        self.score_layer = self.add_weight(shape=(self.hidden_layer_width, 1),
                                           initializer=initializations.get(self.hidden_layer_init),
                                           name='%s_score' % self.name)

    def get_output_shape_for(self, input_shapes):
        # pylint: disable=unused-argument
        return (input_shapes[0][0], 1)

    @overrides
    def compute_mask(self, input, input_mask=None):  # pylint: disable=unused-argument,redefined-builtin
        # Here, input_mask is ignored, because the input is plain word tokens. To determine the returned mask,
        # we want to see if either of the inputs is all padding (i.e. the mask would be all 0s), if so, then
        # the whole tuple_match should be masked, so we would return a 0, otherwise we return a 1.  As such,
        # the shape of the returned mask is (batch size, 1).
        input1, input2 = input
        return K.any(input1) * K.any(input2)

    def call(self, x, mask=None):
        tuple1_input, tuple2_input = x      # tuple1 shape: (batch size, num_slots, num_slot_words_t1)
                                            # tuple2 shape: (batch size, num_slots, num_slot_words_t2)
        # Check that the tuples have the same number of slots.
        assert K.int_shape(tuple1_input)[1] == K.int_shape(tuple2_input)[1]

        # Expand tuple1 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2)
        expanded_tuple1 = K.expand_dims(tuple1_input, 3)    # now (b, num_slots, num_slot_words_tuple1, 1)
        tiled_tuple1 = K.repeat_elements(expanded_tuple1, K.int_shape(tuple2_input)[2], axis=3)

        # Expand tuple2 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2)
        expanded_tuple2 = K.expand_dims(tuple2_input, 2)    # now (b, num_slots, 1, num_slot_words_tuple2)
        tiled_tuple2 = K.repeat_elements(expanded_tuple2, K.int_shape(tuple1_input)[2], axis=2)

        # This generates a binary tensor of the same shape as tiled_tuple1 /
        # tiled_tuple2 that indicates if given word matches between tuple1 and tuple2 in a particular slot.
        # Currently, we only consider S_t1 <--> S_t2 etc overlap, not across slot types.
        # shape: (batch size, num_slots, num_slot_words_tuple1, num_slot_words_tuple2)
        tuple_words_overlap = K.cast(K.equal(tiled_tuple1, tiled_tuple2), "float32")

        # Exclude zeros (i.e. padded elements) from matching each other.
        # tuple1_mask is 1 if tuple1 has a real element, 0 if it's a padding element.
        tiled_tuple1_mask = K.cast(K.not_equal(tiled_tuple1, K.zeros_like(tiled_tuple1, dtype='int32')),
                                   dtype='float32')
        zeros_excluded_overlap = tuple_words_overlap * tiled_tuple1_mask

        # Find non-padding elements in tuple1.
        # shape: (batch size, num_slots, num_slot_words_tuple1)
        non_padded_tuple1 = K.cast(K.not_equal(tuple1_input, K.zeros_like(tuple1_input)), 'float32')
        # Count these non-padded elements to know how many words were in each slot of tuple1.
        # shape: (batch size, num_slots)
        num_tuple1_words_in_each_slot = K.sum(non_padded_tuple1, axis=2)

        # Find the number of words that overlap in each of the slots.
        # shape: (batch size, num_slots)
        slot_overlap_sums = K.sum(K.sum(zeros_excluded_overlap, axis=3), axis=2)

        # # Normalize by the number of words in tuple1.
        # TODO(becky): should this be fixed to tuple1 or allowed to vary? Does switching input order work
        # for varying?
        # This block of code prevents dividing by zero during normalization:
        divisor = num_tuple1_words_in_each_slot
        # If the divisor is zero at a position, we add epsilon to it.
        is_zero_divisor = K.equal(divisor, K.zeros_like(divisor))
        divisor = switch(is_zero_divisor, K.ones_like(divisor) * K.epsilon(), divisor)

        # shape: (batch size, num_slots)
        normalized_slot_overlap = slot_overlap_sums / divisor
        # shape: (batch size, hidden_layer_width)
        raw_entailment = apply_feed_forward(normalized_slot_overlap, self.hidden_layer_weights,
                                            activations.get(self.hidden_layer_activation))
        # shape: (batch size, 1)
        final_score = activations.get(self.final_activation)(K.dot(raw_entailment, self.score_layer))

        return final_score
