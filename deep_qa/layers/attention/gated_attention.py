from keras import backend as K
from overrides import overrides

from ..masked_layer import MaskedLayer
from ...common.checks import ConfigurationError
from ...tensors.backend import switch

GATING_FUNCTIONS = ["*", "+", "||"]


class GatedAttention(MaskedLayer):
    r"""
    This layer implements the majority of the Gated Attention module described in
    `"Gated-Attention Readers for Text Comprehension" by Dhingra et. al 2016
    <https://arxiv.org/pdf/1606.01549.pdf>`_.

    The module is described in section 3.2.2. For each token :math:`d_i` in :math:`D`,
    the GA module forms a "token-specific representation" of the query :math:`q_i` using
    soft attention, and then multiplies the query representation element-wise with the document
    token representation.

        - 1. :math:`\alpha_i = softmax(Q^T d_i)`
        - 2. :math:`q_i = Q \alpha_i`
        - 3. :math:`x_i = d_i \odot q_i` (:math:`\odot` is element-wise multiplication)

    This layer implements equations 2 and 3 above but in a batched manner to get
    :math:`X`, a tensor with all :math:`x_i`. Thus, the input
    to the layer is :math:`\alpha` (``normalized_qd_attention``), a tensor with
    all :math:`\alpha_i`, as well as :math:`Q` (``question_matrix``), and
    :math:`D` (``document_matrix``), a tensor with all :math:`d_i`. Equation 6 uses
    element-wise multiplication to model the interactions between :math:`d_i` and :math:`q_i`,
    and the paper reports results when using other such gating functions like sum or
    concatenation.

    Inputs:
        - ``document_``, a matrix of shape ``(batch, document length, biGRU hidden length)``.
          Represents the document as encoded by the biGRU.
        - ``question_matrix``, a matrix of shape ``(batch, question length, biGRU hidden length)``.
          Represents the question as encoded by the biGRU.
        - ``normalized_qd_attention``, the soft attention over the document and question.
          Matrix of shape ``(batch, document length, question length)``.

    Output:
        - ``X``, a tensor of shape ``(batch, document length, biGRU hidden length)`` if the
          gating function is ``*`` or ``+``, or ``(batch, document length, biGRU hidden length * 2)``
          if the gating function is ``||``  This serves as a representation of each token in
          the document.

    Parameters
    ----------
    gating_function : string, default="*"
        The gating function to use for modeling the interactions between the document and
        query token. Supported gating functions are ``"*"`` for elementwise multiplication,
        ``"+"`` for elementwise addition, and ``"||"`` for concatenation.

    Notes
    -----
    To find out how we calculated equation 1, see the GatedAttentionReader model (roughly,
    a ``masked_batch_dot`` and a ``masked_softmax``)
    """
    def __init__(self, gating_function="*", **kwargs):
        # We need to wait until below to actually handle this, because self.name gets set in
        # super.__init__.
        # allowed gating functions are "*" (multiply), "+" (sum), and "||" (concatenate)
        self.gating_function = gating_function
        if self.gating_function not in GATING_FUNCTIONS:
            raise ConfigurationError("Invalid gating function "
                                     "{}, expected one of {}".format(self.gating_function,
                                                                     GATING_FUNCTIONS))
        super(GatedAttention, self).__init__(**kwargs)

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        return mask[0]

    @overrides
    def compute_output_shape(self, input_shapes):
        return (input_shapes[0][0], input_shapes[0][1], input_shapes[0][2])

    @overrides
    def call(self, inputs, mask=None):
        # document_matrix is of shape (batch, document length, biGRU hidden length).
        # question_matrix is of shape (batch, question length, biGRU hidden length).
        # normalized_qd_attention is of shape (batch, document length, question length).
        document_matrix, question_matrix, normalized_qd_attention = inputs
        if mask is None:
            document_mask = K.ones_like(document_matrix)[:, :, 0]
        else:
            document_mask = mask[0]

        # question_update is of shape (batch, document length, bigru hidden).
        question_update = K.batch_dot(normalized_qd_attention, question_matrix, axes=[2, 1])

        # We use the gating function to calculate the new document representation
        # which is of shape (batch, document length, biGRU hidden length).
        masked_representation = None
        if self.gating_function == "||":
            # shape (batch, document length, biGRU hidden length*2)
            unmasked_representation = K.concatenate([question_update, document_matrix])
            # Apply the mask from the document to zero out things that should be masked.
            # The mask is of shape (batch, document length), so we tile it to
            # shape (batch, document length, biGRU hidden length*2)
            tiled_mask = K.repeat_elements(K.expand_dims(document_mask, axis=2),
                                           (2*K.int_shape(document_matrix)[2]), 2)
            masked_representation = switch(tiled_mask, unmasked_representation,
                                           K.zeros_like(unmasked_representation))
            return masked_representation

        if self.gating_function == "*":
            unmasked_representation = question_update * document_matrix

        if self.gating_function == "+":
            # shape (batch, document length, biGRU hidden length)
            unmasked_representation = question_update + document_matrix

        # Apply the mask from the document to zero out things that should be masked.
        # The mask is of shape (batch, document length), so we tile it to
        # shape (batch, document length, biGRU hidden length)
        tiled_mask = K.repeat_elements(K.expand_dims(document_mask, axis=2),
                                       K.int_shape(document_matrix)[2], 2)
        masked_representation = switch(tiled_mask, unmasked_representation, K.zeros_like(unmasked_representation))

        if masked_representation is not None:
            return masked_representation
        else:
            raise ConfigurationError("Invalid gating function "
                                     "{}, expected one of {}".format(self.gating_function,
                                                                     GATING_FUNCTIONS))

    @overrides
    def get_config(self):
        config = {'gating_function': self.gating_function}
        base_config = super(GatedAttention, self).get_config()
        config.update(base_config)
        return config
