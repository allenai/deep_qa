from typing import Dict
import tensorflow as tf

from keras import backend as K
from keras.regularizers import l1

from .masked_layer import MaskedLayer


class AdaptiveRecurrence:
    '''
    This recurrence class peforms an adaptive number of memory network steps. This is performed
    in the AdaptiveStep layer. In order to take an adaptive number of steps, at each memory network
    iteration, we compute a probability of halting via the dot product of the memory representation
    and a parameter of the model. We are effectively implementing a single step of the Adaptive
    Computation Time algorithm (Graves, 2016): https://arxiv.org/abs/1603.08983. The behaviour of
    this recurrence is governed by the epsilon, max_computation and ponder_cost parameters, described
    below.

    Note that within this Layer, there is tensorflow code, which
    therefore creates a dependency on using the tensorflow Keras backend.

    Using this method of recurrence means building all the layers we use in the memory network step
    _within_ the AdaptiveStep layer. This means that it is difficult to debug using Keras or the
    debugging code within this repository, as all the parameters for intermediate layers will
    be assigned to the AdaptiveLayer.

    Additionally, because Tensorflow builds static computational graphs, this recurrence
    method only supports Layers within it's adaptive step that share weights across memory network steps.
    The reason for this is because we don't know the number of steps we are going to do in advance,
    Tensorflow can't generate all the required weights. However, if weight sharing is used, Tensorflow
    can create one copy of the weights and although the number of iterations of the memory step is
    non-deterministic, the computational graph can be statically defined - we just loop over a part of it.
    '''
    # pylint: disable=unused-argument
    def __init__(self, memory_network, params: Dict=None, **kwargs):
        if params is None:
            params = {}
        self.memory_network = memory_network
        self.adaptive_step_params = params

    def __call__(self, encoded_question, current_memory, encoded_knowledge):
        adaptive_layer = AdaptiveStep(self.memory_network, **self.adaptive_step_params)
        return adaptive_layer([encoded_question, current_memory, encoded_knowledge])


class AdaptiveStep(MaskedLayer):
    '''
    This layer implements a single step of the halting component of the Adaptive Computation Time algorithm,
    generalised so that it can be applied to any arbitrary function. Here, that function is a single memory network
    step. This can be seen as a differentiable while loop, where the halting condition is an accumulated
    'halting probability' which is computed at every memory network step using the following formula:

    halting_probability = sigmoid(W * memory_vector + b)

    where W,b are parameters of the network. This halting probability is then accumulated and when it
    increments over 1 - epsilon, we halt and return the current_memory and the attended_knowledge from
    the last step.

    The main machinery implemented here is to deal with doing this process with batched inputs.
    There is a subtlety here regarding the batch_size, as clearly we will have samples halting
    at different points in the batch. This is dealt with using logical masks to protect accumulated
    probabilities, states and outputs from a timestep t's contribution if they have already reached
    1 - epsilon at a timestep s < t.
    '''
    def __init__(self,
                 memory_network,
                 epsilon: float=0.01,
                 max_computation: int=10,
                 ponder_cost_strength: float=0.05,
                 initialization='glorot_uniform', name='adaptive_layer', **kwargs):
        # Dictates the value at which we halt the memory network steps (1 - epsilon).
        # Necessary so that the network can learn to halt after one step. If we didn't have
        # this, the first halting value is < 1 in practise as it is the output of a sigmoid.
        self.epsilon = epsilon
        self.one_minus_epsilon = tf.constant(1.0 - self.epsilon, name='one_minus_epsilon')
        # Used to bound the number of memory network hops we do. Necessary to prevent
        # the network from learning that the loss it achieves can be minimised by
        # simply not stopping.
        self.max_computation = tf.constant(max_computation, tf.float32, name='max_computation')
        # Regularisation coefficient for the ponder cost. In order to dictate how many steps you want
        # to take, we add |number of steps| to the training objective, in the same way as you might add
        # weight regularisation. This makes the model optimise performance whilst moderating the number
        # of steps it takes. This parameter is _extremely_ sensitive. Consider as well that this parameter
        # will affect the training time of your model, as it will take more steps if it is small. Bear this
        # in mind when doing grid searches over this parameter.
        self.ponder_cost_strength = ponder_cost_strength
        self.memory_network = memory_network
        self.init = initialization
        self.name = name
        # Attributes to be defined when we build this layer.
        self.halting_weight = None
        self.halting_bias = None
        self.trainable_weights = []
        super(AdaptiveStep, self).__init__(**kwargs)

    def build(self, input_shape):
        # pylint: disable=protected-access
        '''
        The only weight that this layer requires is used in a simple dot product with the current_memory
        to generate the halting_probability. We define the weight shape with the 2nd input to this
        layer, as this is the memory representation, which will dictate the required size. Note that this is
        actually building layers defined within the Memory Network. We alter the size of the shapes for each
        of the layers as they are done sequentially and would normally be built with the shape of the input.
        Here, we have to specify it manually as we want to build them in advance, rather than when they are
        called.
        '''
        input_dim = input_shape[1][-1]

        self.halting_weight = self.add_weight(((input_dim,) + (1,)),
                                              initializer=self.init,
                                              name='{}_halting_weight'.format(self.name))
        self.halting_bias = self.add_weight((),
                                            initializer=self.init,
                                            name='{}_halting_bias'.format(self.name))
        self.trainable_weights = [self.halting_weight, self.halting_bias]

        background_knowledge_shape = list(input_shape[2])

        knowledge_selector_input_shape = list(background_knowledge_shape)
        # Shape after appending original question and
        knowledge_selector_input_shape[-2] += 2
        knowledge_selector = self.memory_network._get_knowledge_selector(0)
        knowledge_selector.build(tuple(knowledge_selector_input_shape))
        self.trainable_weights.extend(knowledge_selector.trainable_weights)

        knowledge_combiner_input_shape = list(background_knowledge_shape)
        # Shape after appending the attention mask to the background knowledge.
        knowledge_combiner_input_shape[-1] += 1
        knowledge_combiner = self.memory_network._get_knowledge_combiner(0)
        knowledge_combiner.build(tuple(knowledge_combiner_input_shape))
        self.trainable_weights.extend(knowledge_combiner.trainable_weights)

        memory_updater_input_shape = list(background_knowledge_shape)
        # Shape after removing the knowledge_length dimension (post knowledge_combiner)
        # and concatenating the original_question, current_memory and attended_knowledge.
        memory_updater_input_shape.pop(-2)
        memory_updater_input_shape[-1] *= 3
        memory_updater = self.memory_network._get_memory_updater(0)
        memory_updater.build(tuple(memory_updater_input_shape))
        self.trainable_weights.extend(memory_updater.trainable_weights)
        super(AdaptiveStep, self).build(input_shape)

    def call(self, inputs, mask=None):
        encoded_question, current_memory, encoded_knowledge = inputs
        # We need to create a tensor which doesn't have the encoding_dim dimension. So that this Layer is
        # independent of the dimension of the input tensors, we just sum over the last dimension to remove it.
        # We only use this to create variables, nothing else.
        memory_cell = K.sum(current_memory, -1)
        # This is a boolean mask, holding whether a particular sample has halted.
        batch_mask = tf.cast(tf.ones_like(memory_cell, name='batch_mask'), tf.bool)
        # This counts the number of memory steps per sample.
        hop_counter = tf.zeros_like(memory_cell, name='hop_counter')
        # This accumulates the halting probabilities.
        halting_accumulator = tf.zeros_like(memory_cell, name='halting_accumulator')
        # This also accumulates the halting probabilities, with the difference being that if an
        # outputed probability causes a particular sample to go over 1 - epsilon, this accumulates
        # that value, but the halting_accumulator does not. This variable is _only_ used in the
        # halting condition of the loop.
        halting_accumulator_for_comparison = tf.zeros_like(memory_cell,
                                                           name='halting_acc_for_comparision')
        # This accumulates the weighted memory vectors at each memory step. The memory is weighted by the
        # halting probability and added to this accumulator.
        memory_accumulator = tf.zeros_like(current_memory, name='memory_accumulator')
        # We need the attended_knowledge from the last memory network step, so we create a dummy variable to
        # input to the while_loop, as tensorflow requires the input signature to match the output signature.
        attended_knowledge_loop_placeholder = tf.zeros_like(current_memory, name='attended_knowledge_placeholder')

        # Add the ponder cost variable as a regulariser to the loss function.
        ponder_cost = l1(self.ponder_cost_strength)
        self.add_loss(ponder_cost(hop_counter))
        # This actually does the computation of self.adaptive_memory_hop,
        # checking the condition at every step to see if it should stop.

        # The while loop has to produce as many variables as it has inputs - we only need the last two.
        *_, current_memory, attended_knowledge = \
            tf.while_loop(cond=self.halting_condition, body=self.adaptive_memory_hop,
                          loop_vars=[batch_mask,
                                     halting_accumulator,
                                     halting_accumulator_for_comparison,
                                     hop_counter,
                                     encoded_question,
                                     current_memory,
                                     encoded_knowledge,
                                     memory_accumulator,
                                     attended_knowledge_loop_placeholder
                                    ])

        return [current_memory, attended_knowledge]

    def halting_condition(self,
                          batch_mask,
                          halting_accumulator,
                          halting_accumulator_for_comparison,
                          hop_counter,
                          encoded_question,
                          current_memory,
                          encoded_knowledge,
                          memory_accumulator,
                          attended_knowledge_placeholder):
        # Tensorflow requires that we use all of the variables used in the tf.while_loop as inputs to the
        # condition for halting the loop, even though we only actually make use of two of them.
        # pylint: disable=unused-argument

        # This condition checks the batch elementwise to see if any of the accumulated halting
        # probabilities have gone over one_minus_epsilon in the previous iteration.
        probability_condition = tf.less(halting_accumulator_for_comparison, self.one_minus_epsilon)
        # This condition checks the batch elementwise to see if any have taken more steps than the max allowed.
        max_computation_condition = tf.less(hop_counter, self.max_computation)
        # We only stop if both of the above conditions are true....
        combined_conditions = tf.logical_and(probability_condition, max_computation_condition)
        # ... for the entire batch.
        return tf.reduce_any(combined_conditions)

    def adaptive_memory_hop(self,
                            batch_mask,
                            halting_accumulator,
                            halting_accumulator_for_comparison,
                            hop_counter,
                            encoded_question,
                            previous_memory,
                            encoded_knowledge,
                            memory_accumulator,
                            attended_knowledge):
        '''
        In this method, we do a full step of the memory network and generate the probability of halting
        per example, followed by various updates to counters and masks which are used to control the halting
        mechanism for the batch of samples.
        '''
        # First things first: let's actually do a memory network step. This is exactly the same as in the
        # vanilla memory network.
        current_memory, attended_knowledge = self.memory_network.memory_step(
                encoded_question, previous_memory, encoded_knowledge)

        # Here, we are computing the probability that each sample in the batch will halt at this iteration.
        # This outputs a vector of probabilities of shape (samples, ), or (samples, num_options) for memory
        # networks with multiple answer options.
        with tf.variable_scope("halting_calculation"):
            halting_probability = tf.squeeze(tf.sigmoid(
                    K.dot(current_memory, self.halting_weight) + self.halting_bias), [-1])
        # This is where the loop condition variables are controlled, which takes several steps.
        # First, we compute a new batch mask, which will be of size (samples, ). We want there
        # to be 0s where a given sample's adaptive loop should have halted. To check this, we
        # compare element-wise the halting_accumulator plus this iteration's halting probabilities
        # to see if they are less than 1 - epsilon. Additionally, if a given sample had halted at
        # the previous batch, we don't want these to accidentally start again in this iteration,
        # so we also compare to the previous batch_mask using logical and.

        # Example of why we need to protect against the above scenario:
        # If we were at 0.8 and generated a probability of 0.3, which would take us over 1 - epsilon.
        #  We then don't add this to the halting_accumulator, and then in the next iteration, we
        # generate 0.1, which would not take us over the limit, as the halting_accumulator is still
        # at 0.8. However, we don't want to consider this contribution, as we have already halted.
        new_batch_mask = tf.logical_and(
                tf.less(halting_accumulator + halting_probability, self.one_minus_epsilon), batch_mask)

        # Next, we update the halting_accumulator by adding on the halting_probabilities from this
        # iteration, masked by the new_batch_mask. Note that this means that if the halting_probability
        # for a given sample has caused the accumulator to go over 1 - epsilon, we DO NOT update this
        # value in the halting_accumulator. Values in this accumulator can never be over 1 - epsilon.
        new_float_mask = tf.cast(new_batch_mask, tf.float32)
        halting_accumulator += halting_probability * new_float_mask

        # Finally, we update the halting_accumulator_for_comparison, which is only used in
        # the halting condition in the while_loop. Note that here, we are adding on the halting
        # probabilities multiplied by the previous iteration's batch_mask, which means that we
        # DO update samples over 1 - epsilon. This means that we can check in the loop condition
        # to see if all samples are over 1 - epsilon, which means we should halt the while_loop.
        halting_accumulator_for_comparison += halting_probability * tf.cast(batch_mask, tf.float32)

        # This just counts the number of memory network steps we take for each sample.
        # We use this for regularisation - by adding this to the loss function, we can bias
        # the network to take fewer steps.
        hop_counter += new_float_mask

        # This condition checks whether a sample has gone over the permitted number of memory steps.
        counter_condition = tf.less(hop_counter, self.max_computation)

        # If a given sample is under the max number of steps AND not yet halted, we use the "use_probability"
        # value in the conditional below. This option is just accumulating the memory network state, as the
        # output of this whole loop is a weighted sum of the memory representations with respect to the
        # halting probabilities at each step. Additionally, we multiply by the previous batch mask so that
        # if we have previously stopped for a given batch, we never add any more on in a future timestep.

        # The second "use_remainder" option is taken when a given batch should halt, determined by the
        # final_iteration_condition. Instead of using the final halting_probability, we use
        # 1 - _halting_accumulator, due to the  1 - epsilon halting condition, as the final probability
        # also needs to take into account this epsilon value.
        not_final_iteration_condition = tf.expand_dims(tf.logical_and(new_batch_mask, counter_condition), -1)

        use_probability = tf.expand_dims(halting_probability, -1)
        use_remainder = tf.expand_dims(1.0 - halting_accumulator, -1)

        memory_update_weight = tf.where(not_final_iteration_condition, use_probability, use_remainder)
        expanded_batch_mask = tf.expand_dims(tf.cast(batch_mask, tf.float32), -1)
        memory_accumulator += current_memory * memory_update_weight * expanded_batch_mask

        # We have to return all of these values as a requirement of the tf.while_loop. Some of them,
        # we haven't updated, such as the encoded_question and encoded_knowledge.
        return [new_batch_mask,
                halting_accumulator,
                halting_accumulator_for_comparison,
                hop_counter,
                encoded_question,
                current_memory,
                encoded_knowledge,
                memory_accumulator,
                attended_knowledge]

    def compute_mask(self, inputs, mask=None):  # pylint: disable=unused-argument
        # We don't want to mask either of the outputs here, so we return None for both of them.
        return [None, None]

    def compute_output_shape(self, input_shapes):
        # We output two tensors from this layer, the final memory representation and
        # the attended knowledge from the final memory network step. Both have the same
        # shape as the initial memory vector, likely (samples, encoding_dim), which is
        # passed in as the 2nd argument, so we return this shape twice.
        return [input_shapes[1], input_shapes[1]]

    def get_config(self):
        config = {
                # TODO: This won't work when we reload the model.
                'memory_network': self.memory_network.__class__,
                'name': self.name,
                'init': self.init,
                'ponder_cost_strength': self.ponder_cost_strength,
                'epsilon': self.epsilon,
                'max_computation': self.max_computation
        }
        base_config = super(AdaptiveStep, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
