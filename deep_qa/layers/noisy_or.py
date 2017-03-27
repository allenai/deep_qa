from keras import backend as K
from keras.constraints import Constraint
from keras.regularizers import l1_l2
from overrides import overrides

from .masked_layer import MaskedLayer


class BetweenZeroAndOne(Constraint):
    """
    Constrains the weights to be between zero and one
    """
    def __call__(self, p):
        # Clip values less than or equal to zero to epsilon.
        p *= K.cast(p >= 0., K.floatx())
        leaky_zeros_mask = K.epsilon() * K.cast(K.equal(p, K.zeros_like(p)), 'float32')
        p = p + leaky_zeros_mask
        # Clip values greater to 1 to 1.
        p *= K.cast(p <= 1., K.floatx())
        leaky_ones_mask = K.cast(K.equal(p, K.zeros_like(p)), K.floatx())
        p += leaky_ones_mask
        return p


class NoisyOr(MaskedLayer):
    r"""
    This layer takes as input a tensor of probabilities and calculates the
    noisy-or probability across a given axis based on the noisy-or equation:

    - :math:`p(x) = 1 - \prod_{i=1:N}(1 - q * p(x|y_n))`

    where :math`q` is the noise parameter.

    Inputs:
        - probabilities: shape ``(batch, ..., N, ...)``
          Optionally takes a mask of the same shape,
          where N is the number of y's in the above equation
          (i.e. the number of probabilities being combined in the product),
          in the dimension corresponding to the specified axis.

    Output:
        - X: shape ``(batch, ..., ...)``
          The output has one less dimension than the input, and has an
          optional mask of the same shape.  The lost dimension corresponds
          to the specified axis. The output mask is the result of ``K.any()``
          on the input mask, along the specified axis.

    Parameters
    ----------
    axis : int, default=-1
        The axis over which to combine probabilities.

    name : string, default='noisy_or'
        Name of the layer, ued to debug both the layer and its parameter.

    param_init : string, default='uniform'
        The initialization of the noise parameter.

    noise_param_constraint : Keras Constraint, default=None
        Optional, a constraint which would be applied to the noise parameter.
    """
    def __init__(self, axis=-1, name="noisy_or", param_init='uniform', noise_param_constraint=None, **kwargs):
        self.axis = axis
        self.param_init = param_init
        self.name = name
        # The noisy-or equation includes a noise parameter (q) which is learned during training.
        self.noise_parameter = None
        self.noise_param_constraint = noise_param_constraint
        super(NoisyOr, self).__init__(**kwargs)

    def build(self, input_shape):
        # Add the trainable weight variable for the noise parameter.
        self.noise_parameter = self.add_weight(shape=(),
                                               name=self.name + '_noise_param',
                                               initializer=self.param_init,
                                               regularizer=l1_l2(l2=0.001),
                                               constraint=self.noise_param_constraint,
                                               trainable=True)
        super(NoisyOr, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.axis == -1:
            return input_shape[:-1]
        return input_shape[:self.axis - 1] + input_shape[self.axis:]

    @overrides
    def compute_mask(self, inputs, mask=None):
        # pylint: disable=unused-argument
        if mask is not None:
            return K.any(mask, axis=self.axis)
        return None

    def call(self, inputs, mask=None):
        # shape: (batch size, ..., num_probs, ...)
        probabilities = inputs
        if mask is not None:
            probabilities *= K.cast(mask, "float32")

        noisy_probs = self.noise_parameter * probabilities

        # shape: (batch size, ..., num_probs, ...)
        noisy_probs = 1.0 - noisy_probs

        # shape: (batch size, ..., ...)
        probability_product = K.prod(noisy_probs, axis=self.axis)

        return 1.0 - probability_product
