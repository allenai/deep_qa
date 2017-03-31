'''
Word alignment entailment models operate on word level representations, and define alignment
as a function of how well the words in the premise align with those in the hypothesis. These
are different from the encoded sentence entailment models where both the premise and hypothesis
are encoded as single vectors and entailment functions are defined on top of them.

At this point this doesn't quite fit into the memory network setup because the model doesn't
operate on the encoded sentence representations, but instead consumes the word level
representations.
TODO(pradeep): Make this work with the memory network eventually.
'''

from keras import backend as K

from ..masked_layer import MaskedLayer
from ...tensors.masked_operations import masked_softmax, masked_batch_dot
from ...tensors.backend import last_dim_flatten

class WordAlignmentEntailment(MaskedLayer):  # pylint: disable=abstract-method
    '''
    This is an abstract class for word alignment entailment. It defines an _align function.
    '''
    def __init__(self, **kwargs):
        self.input_dim = None
        super(WordAlignmentEntailment, self).__init__(**kwargs)

    @staticmethod
    def _align(source_embedding, target_embedding, source_mask, target_mask, normalize_alignment=True):
        '''
        Takes source and target sequence embeddings and returns a source-to-target alignment weights.
        That is, for each word in the source sentence, returns a probability distribution over target_sequence
        that shows how well each target word aligns (i.e. is similar) to it.

        source_embedding: (batch_size, source_length, embed_dim)
        target_embedding: (batch_size, target_length, embed_dim)
        source_mask: None or (batch_size, source_length, 1)
        target_mask: None or (batch_size, target_length, 1)
        normalize_alignment (bool): Will apply a (masked) softmax over alignments is True.

        Returns:
        s2t_attention: (batch_size, source_length, target_length)
        '''
        source_dot_target = masked_batch_dot(source_embedding, target_embedding, source_mask, target_mask)
        if normalize_alignment:
            alignment_shape = K.shape(source_dot_target)
            flattened_products_with_source = last_dim_flatten(source_dot_target)
            if source_mask is None and target_mask is None:
                flattened_s2t_attention = K.softmax(flattened_products_with_source)
            elif source_mask is not None and target_mask is not None:
                float_source_mask = K.cast(source_mask, 'float32')
                float_target_mask = K.cast(target_mask, 'float32')
                # (batch_size, source_length, target_length)
                s2t_mask = K.expand_dims(float_source_mask, axis=-1) * K.expand_dims(float_target_mask, axis=1)
                flattened_s2t_mask = last_dim_flatten(s2t_mask)
                flattened_s2t_attention = masked_softmax(flattened_products_with_source, flattened_s2t_mask)
            else:
                # One of the two inputs is masked, and the other isn't. How did this happen??
                raise NotImplementedError('Cannot handle only one of the inputs being masked.')
            # (batch_size, source_length, target_length)
            s2t_attention = K.reshape(flattened_s2t_attention, alignment_shape)
            return s2t_attention
        else:
            return source_dot_target
