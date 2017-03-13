from copy import deepcopy
from typing import Any, Dict

from keras import backend as K
from keras import initializations, activations
from keras.regularizers import L1L2Regularizer
from keras.layers import Layer
from overrides import overrides

from ...tensors.backend import switch, apply_feed_forward
from ...common.params import get_choice_with_default
from ...tensors.similarity_functions import similarity_functions


class ThresholdTupleMatcher(Layer):
    r"""
    This layer takes as input two tensors corresponding to two tuples, an answer tuple and a background tuple,
    and calculates the degree to which the background tuple `entails` the answer tuple.  Entailment is
    determined by generating a set of entailment features from the tuples (the number of
    entailment_features = number of tuple slots), and then passing these features into a shallow NN to get an
    entailment score.
    Each entailment feature is currently made by comparing the corresponding slots in the two tuples and
    determining the degree of lexical overlap based on a similarity threshold, :math:`\tau` which is learned
    by the model.
        Let :math:`T(B_s, A_s) = \{b_s^i \in B_s : sim(b_s^i, a_s^j) > \tau` for some :math:`a_s^j \in A_s \}`,
        where where :math:`s` is the index of the slot, :math:`A_s` is answer tuple slot :math:`s` and :math:`B_s`
        is background tuple slot :math:`s`, and :math:`\tau` is the current similarity threshold.

    That is, $T(B_s, A_s)$ is the set of words in the background slot which are similar enough to words in
    the answer slot to "count" as overlapping.
    Then,
        :math:`normalized\_overlap_s = \dfrac{|T(B_s, A_s)|}{|A_s|}`

    Inputs:
        - tuple_1_input (the answer tuple), shape ``(batch size, num_slots, num_slot_words_t1, embedding_dim)``,
          and ac orresponding mask of shape (``(batch size, num_slots, num_slot_words_t1)``.
          Here num_slot_words_t1 is the maximum number of words in each of the slots in tuple_1.
        - tuple_2_input (the background_tuple),
          shape ``(batch size, num_slots, num_slot_words_t2, embedding_dim)``, and again a corresponding mask
          of shape (``(batch size, num_slots, num_slot_words_t1)``. As above, num_slot_words_t2 is the
          maximum number of words in each of the slots in tuple_2. This need not match tuple 1.

    Output:
        - entailment score, shape ``(batch, 1)``

    Parameters
    ----------
    - similarity_function_params: Dict[str, Any], default={}
        These parameters get passed to a similarity function (see
        :mod:`deep_qa.tensors.similarity_functions` for more info on what's acceptable).  The default
        similarity function with no parameters is a simple dot product.

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

    """

    def __init__(self, similarity_function: Dict[str, Any]=None, num_hidden_layers: int=1,
                 hidden_layer_width: int=4, initialization: str='glorot_uniform',
                 hidden_layer_activation: str='tanh', final_activation: str='sigmoid', **kwargs):
        self.supports_masking = True
        # Parameters for the shallow neural network
        self.num_hidden_layers = num_hidden_layers
        self.hidden_layer_width = hidden_layer_width
        self.hidden_layer_init = initialization
        self.hidden_layer_activation = hidden_layer_activation
        self.final_activation = final_activation
        self.hidden_layer_weights = []
        self.score_layer = None
        # This thresholded matcher includes a similarity threshold which is learned during training.
        self.similarity_threshold = None
        super(ThresholdTupleMatcher, self).__init__(**kwargs)
        self.similarity_function_params = deepcopy(similarity_function)
        if similarity_function is None:
            similarity_function = {}
        sim_function_choice = get_choice_with_default(similarity_function,
                                                      'type',
                                                      list(similarity_functions.keys()))
        similarity_function['name'] = self.name + '_similarity_function'
        self.similarity_function = similarity_functions[sim_function_choice](**similarity_function)

    def get_config(self):
        base_config = super(ThresholdTupleMatcher, self).get_config()
        config = {'similarity_function': self.similarity_function_params,
                  'num_hidden_layers': self.num_hidden_layers,
                  'hidden_layer_width': self.hidden_layer_width,
                  'initialization': self.hidden_layer_init,
                  'hidden_layer_activation': self.hidden_layer_activation,
                  'final_activation': self.final_activation}
        config.update(base_config)
        return config

    def build(self, input_shape):
        super(ThresholdTupleMatcher, self).build(input_shape)
        # Add the parameter for the similarity threshold
        self.similarity_threshold = self.add_weight(shape=(),
                                                    name=self.name + '_similarity_thresh',
                                                    initializer=self.hidden_layer_init,
                                                    regularizer=L1L2Regularizer(l2=0.001),
                                                    trainable=True)

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
        # Here, we want to see if either of the inputs is all padding (i.e. the mask would be all 0s).
        # If so, then the whole tuple_match should be masked, so we would return a 0, otherwise we
        # return a 1.  As such, the shape of the returned mask is (batch size, 1).
        if input_mask == [None, None]:
            return None
        # Each of the two masks in input_mask are of shape: (batch size, num_slots, num_slot_words)
        mask1, mask2 = input_mask
        mask = K.any(mask1, axis=[1, 2]) * K.any(mask2, axis=[1, 2])
        return K.expand_dims(mask)

    def get_output_mask_shape_for(self, input_shape):  # pylint: disable=no-self-use
        # input_shape is [(batch_size, num_slots, num_slot_words_t1, embedding_dim),
        # (batch_size, num_slots, num_slot_words_t2, embedding_dim)]
        mask_shape = (input_shape[0][0], 1)
        return mask_shape

    def call(self, x, mask=None):
        tuple1_input, tuple2_input = x    # tuple1 shape: (batch size, num_slots, num_slot_words_t1, embedding_dim)
                                          # tuple2 shape: (batch size, num_slots, num_slot_words_t2, embedding_dim)
        # Check that the tuples have the same number of slots.
        assert K.int_shape(tuple1_input)[1] == K.int_shape(tuple2_input)[1]
        num_slot_words_t1 = K.int_shape(tuple1_input)[2]
        num_slot_words_t2 = K.int_shape(tuple2_input)[2]

        # Expand tuple1 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2, embedding_dim)
        # First expand to shape: (batch size, num_slots, num_slot_words_tuple1, 1, embedding_dim)
        expanded_tuple1 = K.expand_dims(tuple1_input, 3)
        # Tile to desired dimensions.
        tiled_tuple1 = K.tile(expanded_tuple1, [1, 1, 1, num_slot_words_t2, 1])

        # Expand tuple2 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2, embedding_dim)
        # First expand to shape: (batch size, num_slots, 1, num_slot_words_tuple2, embedding_dim)
        expanded_tuple2 = K.expand_dims(tuple2_input, 2)
        # Tile to desired dimensions.
        tiled_tuple2 = K.tile(expanded_tuple2, [1, 1, num_slot_words_t2, 1, 1])

        # Generate the similarity scores of each of the word pairs.
        # shape: (batch size, num_slots, num_slot_words_tuple1, num_slot_words_tuple2)
        tuple_word_similarities = self.similarity_function.compute_similarity(tiled_tuple1, tiled_tuple2)

        # This generates a binary tensor of the same shape as tiled_tuple1 /
        # tiled_tuple2 that indicates if given word in one tuple has a high enough similarity to the corresponding
        # word in the other tuple, in a particular slot.
        # Currently, we only consider SUBJ_t1 <--> SUBJ_t2 etc similarities, not across slot types.
        # shape: (batch size, num_slots, num_slot_words_tuple1, num_slot_words_tuple2)
        # TODO(becky): this isn't actually differentiable, is it?  fix?? don't care??
        tuple_words_overlap = K.cast(tuple_word_similarities >= self.similarity_threshold, "float32")

        # Exclude padded/masked elements from counting.
        zeros_excluded_overlap = tuple_words_overlap
        # First, tuple1:
        if mask[0] is not None:
            # Originally, masks for tuples are of shape: (batch size, num_slots, num_slot_words)
            # Expand mask for tuple1 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2)
            expanded_tuple1_mask = K.expand_dims(K.cast(mask[0], "float32"), 3)
            tiled_tuple1_mask = K.cast(K.tile(expanded_tuple1_mask, [1, 1, 1, num_slot_words_t2]), "float32")
            # Do the exclusion.
            zeros_excluded_overlap *= tiled_tuple1_mask
        # Next, tuple 2:
        if mask[1] is not None:
            # Expand mask for tuple2 to shape: (batch size, num_slots, num_slot_words_t1, num_slot_words_t2)
            expanded_tuple2_mask = K.expand_dims(K.cast(mask[1], "float32"), 2)
            tiled_tuple2_mask = K.cast(K.tile(expanded_tuple2_mask, [1, 1, num_slot_words_t1, 1]), "float32")
            # Do the exclusion.
            zeros_excluded_overlap *= tiled_tuple2_mask

        # Find non-padding elements in tuple1.
        # shape: (batch size, num_slots, num_slot_words_tuple1)
        if mask[0] is not None:
            # Mask shape: (batch size, num_slots, num_slot_words_tuple1)
            # Count the non-padded elements to know how many words were in each slot of tuple1.
            # shape: (batch size, num_slots)
            num_tuple1_words_in_each_slot = K.sum(K.cast(mask[0], "float32"), axis=2)
        else:
            # If there's no mask, we can assume that all the words are valid, so we just need to count them.
            # Get a tensor of shape == tuple1_input but without the last dimension.  Do this by summing.
            # shape: (batch size, num_slots, num_slot_words_tuple1)
            reshaped = K.sum(tuple1_input, axis=-1)
            # Now, count.
            # shape: (batch size, num_slots)
            num_tuple1_words_in_each_slot = K.sum(K.ones_like(reshaped, dtype="float32"), axis=2)

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
