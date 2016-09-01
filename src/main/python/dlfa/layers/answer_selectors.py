from keras import backend as K
from keras.layers import Layer

class ArgmaxAnswerSelector(Layer):
    def __init__(self, num_options, name="answer_selector"):
        super(ArgmaxAnswerSelector, self).__init__(name=name)
        self.num_options = num_options

    def call(self, x, mask=None):
        """
        x is (batch_size, num_options, 2), the result of the entailment model on each of the answer
        options.  We want to select among them to get a vector of (batch_size, num_options), which
        will be the final model output.  Here we're just going to look at the true probability from
        the entailment model, and do an argmax to get the vector of (batch_size, num_options).
        """
        entailment_prob = x[:, :, 1]
        best_option = K.argmax(entailment_prob, axis=1)

        # TODO(matt): this does not work!  It's not differentiable, so we can't train the model...
        return K.one_hot(best_option, self.num_options)

    def get_output_shape_for(self, input_shape):
        return input_shape[:2]
