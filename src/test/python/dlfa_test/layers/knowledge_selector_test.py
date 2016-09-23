import numpy
from keras import initializations
from keras import backend as K

from dlfa.layers.knowledge_selectors import hardmax

class TestKnowledgeSelector:
    def test_hardmax(self):
        num_samples = 10
        knowledge_length = 5
        init = initializations.get('uniform')
        unnormalized_attention = init((num_samples, knowledge_length))
        hardmax_output = hardmax(unnormalized_attention, knowledge_length)
        input_value = K.eval(unnormalized_attention)
        output_value = K.eval(hardmax_output)
        assert output_value.shape == (num_samples, knowledge_length)
        # Assert all elements other than the ones are zeros
        assert numpy.count_nonzero(output_value) == num_samples
        # Assert the max values in all rows are ones
        assert numpy.all(numpy.equal(numpy.max(output_value, axis=1),
                                     numpy.ones((num_samples,))))
        # Assert ones are in the right places
        assert numpy.all(numpy.equal(numpy.argmax(output_value, axis=1),
                                     numpy.argmax(input_value, axis=1)))
