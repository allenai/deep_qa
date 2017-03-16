from keras import backend as K
from keras.engine import InputSpec
from keras.layers import Layer, Convolution1D, MaxPooling1D, merge, Dense
from keras.regularizers import l1l2

class CNNEncoder(Layer):
    '''
    CNNEncoder is a combination of multiple convolution layers and max pooling layers. This is
    defined as a single layer to be consistent with the other encoders in terms of input and output
    specifications.  The input to this "layer" is of shape (batch_size, num_words, embedding_dim)
    and the output is of size (batch_size, output_dim).

    The CNN has one convolution layer per each ngram filter size. Each convolution operation gives
    out a vector of size num_filters. The number of times a convolution layer will be used
    depends on the ngram size: input_length - ngram_size + 1. The corresponding maxpooling layer
    aggregates all these outputs from the convolution layer and outputs the max.

    This operation is repeated for every ngram size passed, and consequently the dimensionality of
    the output after maxpooling is len(ngram_filter_sizes) * num_filters.

    We then use a fully connected layer to project in back to the desired output_dim.  For more
    details, refer to "A Sensitivity Analysis of (and Practitioners’ Guide to) Convolutional Neural
    Networks for Sentence Classification", Zhang and Wallace 2016, particularly Figure 1.
    '''
    def __init__(self, weights=None, **kwargs):
        self.supports_masking = True

        # This is the output dim for each convolutional layer, which is the same as the number of
        # "filters" learned by that layer.
        self.num_filters = kwargs.pop('num_filters')

        # This specifies both the number of convolutional layers we will create and their sizes.
        # Must be a List[int].  The default of (2, 3, 4, 5) will have four convolutional layers,
        # corresponding to encoding ngrams of size 2 to 5 with some number of filters.
        ngram_filter_sizes = kwargs.pop('ngram_filter_sizes', (2, 3, 4, 5))
        self.ngram_filter_sizes = ngram_filter_sizes

        self.output_dim = kwargs.pop('output_dim')

        conv_layer_activation = kwargs.pop('conv_layer_activation', 'relu')
        self.conv_layer_activation = conv_layer_activation

        self.l1_regularization = kwargs.pop("l1_regularization", None)
        self.l2_regularization = kwargs.pop("l2_regularization", None)
        self.regularizer = lambda: l1l2(l1=self.l1_regularization, l2=self.l2_regularization)

        # These are member variables that will be defined during self.build().
        self.convolution_layers = None
        self.max_pooling_layers = None
        self.projection_layer = None

        self.input_spec = [InputSpec(ndim=3)]
        self.initial_weights = weights
        super(CNNEncoder, self).__init__(**kwargs)

    def build(self, input_shape):
        input_length = input_shape[1]  # number of words
        # We define convolution, maxpooling and dense layers first.
        self.convolution_layers = [Convolution1D(nb_filter=self.num_filters,
                                                 filter_length=ngram_size,
                                                 activation=self.conv_layer_activation,
                                                 W_regularizer=self.regularizer(),
                                                 b_regularizer=self.regularizer())
                                   for ngram_size in self.ngram_filter_sizes]
        self.max_pooling_layers = [MaxPooling1D(pool_length=input_length - ngram_size + 1)
                                   for ngram_size in self.ngram_filter_sizes]
        self.projection_layer = Dense(self.output_dim)
        # Building all layers because these sub-layers are not explitly part of the computatonal graph.
        for convolution_layer, max_pooling_layer in zip(self.convolution_layers, self.max_pooling_layers):
            convolution_layer.build(input_shape)
            max_pooling_layer.build(convolution_layer.get_output_shape_for(input_shape))
        maxpool_output_dim = self.num_filters * len(self.ngram_filter_sizes)
        projection_input_shape = (input_shape[0], maxpool_output_dim)
        self.projection_layer.build(projection_input_shape)
        # Defining the weights of this "layer" as the set of weights from all convolution
        # and maxpooling layers.
        self.trainable_weights = []
        for layer in self.convolution_layers + self.max_pooling_layers + [self.projection_layer]:
            self.trainable_weights.extend(layer.trainable_weights)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        super(CNNEncoder, self).build(input_shape)

    def call(self, x, mask=None):
        # Each convolution layer returns output of size (samples, pool_length, num_filters),
        #       where pool_length = num_words - ngram_size + 1
        # Each maxpooling layer returns output of size (samples, 1, num_filters).
        # We need to flatten to remove the second dimension of length 1 from the maxpooled output.
        filter_outputs = [K.batch_flatten(max_pooling_layer.call(convolution_layer.call(x, mask)))
                          for max_pooling_layer, convolution_layer in zip(self.max_pooling_layers,
                                                                          self.convolution_layers)]
        maxpool_output = merge(filter_outputs, mode='concat') if len(filter_outputs) > 1 else filter_outputs[0]
        return self.projection_layer.call(maxpool_output)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def compute_mask(self, input, input_mask=None):  # pylint: disable=redefined-builtin
        # By default Keras propagates the mask from a layer that supports masking. We don't need it
        # anymore. So eliminating it from the flow.
        return None

    def get_config(self):
        config = {"output_dim": self.output_dim,
                  "num_filters": self.num_filters,
                  "ngram_filter_sizes": self.ngram_filter_sizes,
                  "conv_layer_activation": self.conv_layer_activation,
                  "l1_regularization": self.l1_regularization,
                  "l2_regularization": self.l2_regularization,
                 }
        base_config = super(CNNEncoder, self).get_config()
        config.update(base_config)
        return config
