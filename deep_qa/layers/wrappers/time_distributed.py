from keras import backend as K
from keras.layers import InputSpec, TimeDistributed as KerasTimeDistributed
from overrides import overrides

class TimeDistributed(KerasTimeDistributed):
    """
    This class fixes two bugs in Keras: (1) the input mask is not passed to the wrapped layer, and
    (2) Keras' TimeDistributed currently only allows a single input, not a list.  We currently
    don't handle the case where the _output_ of the wrapped layer is a list, however.  (Not that
    that's particularly hard, we just haven't needed it yet, so haven't implemented it.)

    Notes
    -----
    If the output shape for TimeDistributed has a final dimension of 1, we essentially sqeeze it,
    reshaping to have one fewer dimension.  That change takes place in the actual ``call`` method as well as
    the ``compute_output_shape`` method.
    """
    def __init__(self, layer, keep_dims=False, **kwargs):
        self.keep_dims = keep_dims
        super(TimeDistributed, self).__init__(layer, **kwargs)

    @overrides
    def build(self, input_shape):
        if isinstance(input_shape, tuple):
            input_shape = [input_shape]
        assert all(len(shape) >= 3 for shape in input_shape), "Need 3 dims to TimeDistribute"
        all_timesteps = [i[1] for i in input_shape]
        assert len(set(all_timesteps)) == 1, "Tensors must have same number of timesteps"
        self.input_spec = [InputSpec(shape=shape) for shape in input_shape]
        if not self.layer.built:
            child_input_shape = [(shape[0],) + shape[2:] for shape in input_shape]
            if len(input_shape) == 1:
                child_input_shape = child_input_shape[0]
            self.layer.build(child_input_shape)
            self.layer.built = True
        self.built = True
        # It's important that we call Wrapper.build() here, because it sets some important member
        # variables.  But we can't call KerasTimeDistributed.build(), because it assumes only one
        # input, which we're trying to fix.  So we use super(KerasTimeDistributed, self).build()
        # here on purpose - this is not a copy-paste bug.
        super(KerasTimeDistributed, self).build(input_shape)  # pylint: disable=bad-super-call

    @overrides
    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        child_input_shape = [(shape[0],) + shape[2:] for shape in input_shape]
        timesteps = input_shape[0][1]
        if len(input_shape) == 1:
            child_input_shape = child_input_shape[0]
        child_output_shape = self.layer.compute_output_shape(child_input_shape)
        reshaped_shape = (child_output_shape[0], timesteps) + child_output_shape[1:]
        if reshaped_shape[-1] == 1 and not self.keep_dims:
            reshaped_shape = reshaped_shape[:-1]
        return reshaped_shape

    def get_output_mask_shape_for(self, input_shape):
        if not isinstance(input_shape, list):
            input_shape = [input_shape]
        child_input_shape = [(shape[0],) + shape[2:] for shape in input_shape]
        timesteps = input_shape[0][1]
        if len(input_shape) == 1:
            child_input_shape = child_input_shape[0]
        child_output_shape = self.layer.get_output_mask_shape_for(child_input_shape)
        return (child_output_shape[0], timesteps) + child_output_shape[1:]

    @staticmethod
    def reshape_inputs_and_masks(inputs, masks):
        reshaped_xs = []
        reshaped_masks = []
        for x_i, mask_i in zip(inputs, masks):
            input_shape = K.int_shape(x_i)
            reshaped_x = K.reshape(x_i, (-1,) + input_shape[2:])  # (batch_size * timesteps, ...)
            if mask_i is not None:
                mask_ndim = K.ndim(mask_i)
                input_ndim = K.ndim(x_i)
                if mask_ndim == input_ndim:
                    mask_shape = input_shape
                elif mask_ndim == input_ndim - 1:
                    mask_shape = input_shape[:-1]
                else:
                    raise Exception("Mask is of an unexpected shape. Mask's ndim: %s, input's ndim %s" %
                                    (mask_ndim, input_ndim))
                mask_i = K.reshape(mask_i, (-1,) + mask_shape[2:])  # (batch_size * timesteps, ...)
            reshaped_xs.append(reshaped_x)
            reshaped_masks.append(mask_i)
        if len(inputs) == 1:
            reshaped_xs = reshaped_xs[0]
            reshaped_masks = reshaped_masks[0]
        return reshaped_xs, reshaped_masks

    @overrides
    def call(self, inputs, mask=None):
        # Much of this is copied from the Keras 1.0(ish) version of TimeDistributed, though we've
        # modified it quite a bit, to fix the problems mentioned in the docstring and to use better
        # names.
        if not isinstance(inputs, list):
            inputs = [inputs]
            mask = [mask]
        else:
            if mask is None:
                mask = [None] * len(inputs)
        timesteps = K.int_shape(inputs[0])[1]
        input_shape = [K.int_shape(x_i) for x_i in inputs]
        if len(inputs) == 1:
            input_shape = input_shape[0]
        if len(inputs) == 1 and input_shape[0]:
            # The batch size is passed when defining the layer in some cases (for example if it is
            # stateful).  We respect the input shape in that case and don't reshape the input. This
            # is slower.  K.rnn also expects only a single tensor, so we can't do this if we have
            # multiple inputs.
            inputs = inputs[0]
            mask = mask[0]
            def step(x_i, _):
                output = self.layer.call(x_i)
                return output, []
            _, outputs, _ = K.rnn(step, inputs, mask=mask, initial_states=[])
        else:
            reshaped_xs, reshaped_masks = self.reshape_inputs_and_masks(inputs, mask)
            outputs = self.layer.call(reshaped_xs, mask=reshaped_masks)
            output_shape = self.compute_output_shape(input_shape)
            reshaped_shape = (-1, timesteps) + output_shape[2:]
            if reshaped_shape[-1] == 1 and not self.keep_dims:
                reshaped_shape = reshaped_shape[:-1]
            outputs = K.reshape(outputs, reshaped_shape)
        return outputs

    @overrides
    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        if isinstance(mask, list):
            if not any(mask):
                return None
            else:
                raise RuntimeError("This version of TimeDistributed doesn't handle multiple masked "
                                   "inputs!  Use a subclass of TimeDistributed instead.")
        return mask

    @overrides
    def get_config(self):
        base_config = super(TimeDistributed, self).get_config()
        config = {'keep_dims': self.keep_dims}
        config.update(base_config)
        return config
