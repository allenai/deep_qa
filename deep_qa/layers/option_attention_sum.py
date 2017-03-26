from keras import backend as K
from overrides import overrides

from .masked_layer import MaskedLayer
from ..common.checks import ConfigurationError
from ..tensors.backend import switch


class OptionAttentionSum(MaskedLayer):
    """
    This Layer takes three inputs: a tensor of document indices, a tensor of
    document probabilities, and a tensor of answer options. In addition, it takes a
    parameter: a string describing how to calculate the probability of options
    that consist of multiple words. We compute the probability of each of
    the answer options in the fashion described in the paper "Text
    Comprehension with the Attention Sum Reader Network" (Kadlec et. al 2016).

    Inputs:
        - document indices: shape ``(batch_size, document_length)``
        - document probabilities: shape ``(batch_size, document_length)``
        - options: shape ``(batch size, num_options, option_length)``

    Output:
        - option_probabilities ``(batch_size, num_options)``
    """
    def __init__(self, multiword_option_mode="mean", **kwargs):
        """
        Construct a new OptionAttentionSum layer.

        Parameters
        ----------
        multiword_option_mode: str, optional (default="mean")
            Describes how to calculate the probability of options
            that contain multiple words. If "mean", the probability of
            the option is taken to be the mean of the probabilities of
            its constituent words. If "sum", the probability of the option
            is taken to be the sum of the probabilities of its constituent
            words.
        """

        if multiword_option_mode != "mean" and multiword_option_mode != "sum":
            raise ConfigurationError("multiword_option_mode must be 'mean' or "
                                     "'sum', got {}.".format(multiword_option_mode))
        self.multiword_option_mode = multiword_option_mode
        super(OptionAttentionSum, self).__init__(**kwargs)

    @overrides
    def compute_output_shape(self, input_shapes):
        return (input_shapes[2][0], input_shapes[2][1])

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        options = inputs[2]
        padding_mask = K.not_equal(options, K.zeros_like(options))
        return K.cast(K.any(padding_mask, axis=2), "float32")

    @overrides
    def call(self, inputs, mask=None):
        """
        Calculate the probability of each answer option.

        Parameters
        ----------
        inputs: List of Tensors
            The inputs to the layer must be passed in as a list to the
            ``call`` function. The inputs expected are a Tensor of
            document indices, a Tensor of document probabilities, and
            a Tensor of options (in that order).
            The documents indices tensor is a 2D tensor of shape
            (batch size, document_length).
            The document probabilities tensor is a 2D Tensor of shape
            (batch size, document_length).
            The options tensor is of shape (batch size, num_options,
            option_length).
        mask: Tensor or None, optional (default=None)
            Tensor of shape (batch size, max number of options) representing
            which options are padding and thus have a 0 in the associated
            mask position.

        Returns
        -------
        options_probabilities : Tensor
            Tensor with shape (batch size, max number of options) with floats,
            where each float is the normalized probability of the option as
            calculated based on ``self.multiword_option_mode``.
        """
        document_indices, document_probabilities, options = inputs
        # This takes `document_indices` from (batch_size, document_length) to
        # (batch_size, num_options, option_length, document_length), with the
        # original indices repeated, so that we can create a mask indicating
        # which options are used in the probability computation. We do the
        # same thing for `document_probababilities` to select the probability
        # values corresponding to the words in the options.
        expanded_indices = K.expand_dims(K.expand_dims(document_indices, 1), 1)
        tiled_indices = K.repeat_elements(K.repeat_elements(expanded_indices,
                                                            K.int_shape(options)[1], axis=1),
                                          K.int_shape(options)[2], axis=2)

        expanded_probabilities = K.expand_dims(K.expand_dims(document_probabilities, 1), 1)
        tiled_probabilities = K.repeat_elements(K.repeat_elements(expanded_probabilities,
                                                                  K.int_shape(options)[1], axis=1),
                                                K.int_shape(options)[2], axis=2)

        expanded_options = K.expand_dims(options, 3)
        tiled_options = K.repeat_elements(expanded_options,
                                          K.int_shape(document_indices)[-1], axis=3)

        # This generates a binary tensor of the same shape as tiled_options /
        # tiled_indices that indicates if index is option or padding.
        options_words_mask = K.cast(K.equal(tiled_options, tiled_indices),
                                    "float32")

        # This applies a mask to the probabilities to select the
        # indices for probabilities that correspond with option words.
        selected_probabilities = options_words_mask * tiled_probabilities

        # This sums up the probabilities to get the aggregate probability for
        # each option's constituent words.
        options_word_probabilities = K.sum(selected_probabilities, axis=3)

        sum_option_words_probabilities = K.sum(options_word_probabilities,
                                               axis=2)

        if self.multiword_option_mode == "mean":
            # This block figures out how many words (excluding
            # padding) are in each option.
            # Here we generate the mask on the input option.
            option_mask = K.cast(K.not_equal(options, K.zeros_like(options)),
                                 "float32")
            # This tensor stores the number words in each option.
            divisor = K.sum(option_mask, axis=2)
            # If the divisor is zero at a position, we add epsilon to it.
            is_zero_divisor = K.equal(divisor, K.zeros_like(divisor))
            divisor = switch(is_zero_divisor, K.ones_like(divisor)*K.epsilon(), divisor)
        else:
            # Since we're taking the sum, we divide all sums by 1.
            divisor = K.ones_like(sum_option_words_probabilities)

        # Now we divide the sums by the divisor we generated above.
        option_probabilities = sum_option_words_probabilities / divisor
        return option_probabilities

    @overrides
    def get_config(self):
        config = {'multiword_option_mode': self.multiword_option_mode}
        base_config = super(OptionAttentionSum, self).get_config()
        config.update(base_config)
        return config
