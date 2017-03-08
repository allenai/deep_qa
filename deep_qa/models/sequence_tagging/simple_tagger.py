from typing import Any, Dict

from keras.layers import Dense, Input
from overrides import overrides

from ...common.params import get_choice
from ...data.instances.sequence_tagging import concrete_instances
from ...layers.wrappers.time_distributed import TimeDistributed
from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class SimpleTagger(TextTrainer):
    """
    This ``SimpleTagger`` simply encodes a sequence of text with some number of stacked
    ``seq2seq_encoders``, then predicts a tag at each index.

    Parameters
    ----------
    num_stacked_rnns : int, optional (default: ``1``)
        The number of ``seq2seq_encoders`` that we should stack on top of each other before
        predicting tags.
    instance_type : str
        Specifies the particular subclass of ``TaggedSequenceInstance`` to use for loading data,
        which in turn defines things like how the input data is formatted and tokenized.
    """
    def __init__(self, params: Dict[str, Any]):
        self.num_stacked_rnns = params.pop('num_stacked_rnns', 1)
        instance_type_choice = get_choice(params, "instance_type", concrete_instances.keys())
        self.instance_type = concrete_instances[instance_type_choice]
        super(SimpleTagger, self).__init__(params)

    @overrides
    def _instance_type(self):  # pylint: disable=no-self-use
        return self.instance_type

    @overrides
    def _build_model(self):
        # shape: (batch_size, text_length)
        text_input = Input(shape=self._get_sentence_shape(), dtype='int32', name='text_input')
        # shape: (batch_size, text_length, embedding_size)
        text_embedding = self._embed_input(text_input)
        for i in range(self.num_stacked_rnns):
            encoder = self._get_seq2seq_encoder(name="encoder_{}".format(i),
                                                fallback_behavior="use default params")
            # shape still (batch_size, text_length, embedding_size)
            text_embedding = encoder(text_embedding)
        # The -2 below is because we are ignoring the padding and unknown tokens that the
        # DataIndexer has by default.
        predicted_tags = TimeDistributed(Dense(self.data_indexer.get_vocab_size('tags') - 2,
                                               activation='softmax'))(text_embedding)
        return DeepQaModel(input=text_input, output=predicted_tags)

    @overrides
    def _set_max_lengths_from_model(self):
        # TODO(matt): implement this correctly
        pass
