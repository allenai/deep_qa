from copy import deepcopy
from typing import Dict

from keras.layers import Input
from overrides import overrides

from ...common.models import get_submodel
from ...common.params import Params
from ...data.instances.reading_comprehension import McQuestionPassageInstance
from ...layers.attention import Attention
from ...layers.backend import Envelope
from ...layers.backend import Multiply
from ...layers.wrappers import EncoderWrapper
from ...layers.wrappers import TimeDistributedWithMask
from ...models.reading_comprehension import BidirectionalAttentionFlow
from ...training.models import DeepQaModel
from ...training.text_trainer import TextTrainer


class MultipleChoiceBidaf(TextTrainer):
    """
    This class extends Minjoon Seo's `Bidirectional Attention Flow model
    <https://www.semanticscholar.org/paper/Bidirectional-Attention-Flow-for-Machine-Seo-Kembhavi/7586b7cca1deba124af80609327395e613a20e9d>`_,
    which was originally applied to predicting spans from a passage, to answering multiple choice
    questions.

    The approach we're going to take here is to load a BiDAF model directly (literally taking all
    of the parameters we need to construct the ``BidirectionalAttentionFlow`` model class),
    applying it to a question and passage, and then adding a few layers on top to try to match the
    predicted span to the answer options we have.

    To match the predicted span to the answer options, we'll first constructed a weighted
    representation of the passage, weighted by the likelihood of each word in the passage being a
    part of the span.  Then we'll compare that representation to a representation for each answer
    option.

    Input:
        - a passage of shape ``(batch_size, num_passage_words)``
        - a question of shape ``(batch_size, num_question_words)``
        - a set of answer options of shape ``(batch_size, num_options, num_option_words)``

    Output:
        - a probability distribution over the answer options, of shape ``(batch_size, num_options)``

    Parameters
    ----------
    bidaf_params : Dict[str, Any]
        These parameters get passed to a
        :class:`~deep_qa.models.reading_comprehension.bidirectional_attention.BidirectionalAttentionFlow`
        object, which we load.  They should be exactly the same as the parameters used to train the
        saved model.  There is one parameter that must be consistent across the contained BiDAF
        model and this ``TextTrainer`` object, so we copy that parameter from the BiDAF params,
        overwriting any parameters that you set for this ``MultipleChoiceBidaf`` model.  This
        parameter is "tokenizer".
    train_bidaf : bool, optional (default=``False``)
        Should we optimize the weights in the contained BiDAF model, or just the weights that we
        define here?  TODO(matt): setting this to ``True`` is currently incompatible with saving
        and loading the ``MultipleChoiceBidaf`` model.  Getting that to work is not a high
        priority, as we're assuming you have far less multiple choice data, so you want a smaller
        model, anyway.
    num_options : int, optional (default=``None``)
        For padding.  How many options should we pad the data to?  If ``None``, this is set from
        the data.
    num_option_words : int, optional (default=``None``)
        For padding.  How many words are in each answer option?  If ``None``, this is set from
        the data.
    similarity_function : Dict[str, Any], optional (default={'type': 'bilinear'})
        This is the similarity function used to compare an encoded span representation with encoded
        option representations.  These parameters get passed to a similarity function (see
        :mod:`deep_qa.tensors.similarity_functions` for more info on what's acceptable).  The
        default similarity function with no parameters is a set of linear weights on the
        concatenated inputs.  Note that the inputs to this similarity function will have `different
        sizes`, so the set of functions you can use is constrained (i.e., no dot product, etc.).
        Also note that you almost certainly want to have some kind of bilinear interaction, or
        linear with a hidden layer, or something, because fundamentally we want to say whether two
        vectors are close in some projected space, which can't really be captured by a simple
        linear similarity function.

    Notes
    -----
    Porting the code to Keras 2 made this break for some reason that I haven't been able to figure
    out yet.  I told py.test to skip the test we had for this, so I'm moving it to ``contrib``
    until such time as I get the test to actually pass.
    """
    # pylint: disable=protected-access
    def __init__(self, params: Params):
        bidaf_params = params.pop('bidaf_params')
        params['tokenizer'] = deepcopy(bidaf_params.get('tokenizer', {}))
        self._bidaf_model = BidirectionalAttentionFlow(bidaf_params)
        self._bidaf_model.load_model()
        self.train_bidaf = params.pop('train_bidaf', False)
        self.num_options = params.pop('num_options', None)
        self.num_option_words = params.pop('num_option_words', None)
        self.similarity_function_params = params.pop('similarity_function', {'type': 'bilinear'})
        super(MultipleChoiceBidaf, self).__init__(params)
        self.data_indexer = self._bidaf_model.data_indexer
        # We need to not add any more words to the vocabulary, or the model will crash, because
        # we're using the same embedding layer as BiDAF.  So we finalize the data indexer, which
        # will give us some warnings when we try to fit the indexer to the training data, but won't
        # actually add anything.  Also note that this has to happen _after_ we call the superclass
        # constructor, or self.data_indexer will get overwritten.  TODO(matt): make it so you can
        # expand the embedding size after the fact in a loaded model (though that seems really hard
        # to do correctly, especially in this setting where we're working directly with a loaded
        # Keras model).  An alternative would be to have our own embedding layer that's initialized
        # from BiDAF's, use that, then use BiDAF for the phrase layer...  Either way is pretty
        # complicated.
        self.data_indexer.finalize()

    @overrides
    def _build_model(self):
        """
        Our basic outline here will be to run the BiDAF model on the question and the passage, then
        compute an envelope over the passage for what words BiDAF thought were in the answer span.
        Then we'll weight the BiDAF passage, and use the BiDAF encoder to encode the answer
        options.  Then we'll have a simple similarity function on top to score the similarity
        between each answer option and the predicted answer span.

        Getting the right stuff out of the BiDAF model is a little tricky.  We're going to use the
        same approach as done in :meth:`TextTrainer._build_debug_model
        <deep_qa.training.trainer.TextTrainer._build_debug_model>`: we won't modify the model at
        all, but we'll construct a new model that just changes the outputs to be various layers of
        the original model.
        """
        question_shape = self._bidaf_model._get_sentence_shape(self._bidaf_model.num_question_words)
        question_input = Input(shape=question_shape, dtype='int32', name="question_input")
        passage_shape = self._bidaf_model._get_sentence_shape(self._bidaf_model.num_passage_words)
        passage_input = Input(shape=passage_shape, dtype='int32', name="passage_input")
        options_shape = (self.num_options,) + self._bidaf_model._get_sentence_shape(self.num_option_words)
        options_input = Input(shape=options_shape, dtype='int32', name='options_input')

        # First we compute a span envelope over the passage, then multiply that by the passage
        # representation.
        bidaf_passage_model = get_submodel(self._bidaf_model.model,
                                           ['question_input', 'passage_input'],
                                           ['final_merged_passage', 'span_begin_softmax', 'span_end_softmax'],
                                           train_model=self.train_bidaf,
                                           name="passage_model")
        modeled_passage, span_begin, span_end = bidaf_passage_model([question_input, passage_input])
        envelope = Envelope()([span_begin, span_end])
        weighted_passage = Multiply()([modeled_passage, envelope])

        # Then we encode the answer options the same way we encoded the question.
        bidaf_question_model = get_submodel(self._bidaf_model.model,
                                            ['question_input'],
                                            ['phrase_encoder'],
                                            train_model=self.train_bidaf,
                                            name="phrase_encoder_model")
        # Total hack to make this compatible with TimeDistributedWithMask.  Ok, ok, python's duck
        # typing is kind of nice sometimes...  At least I can get this to work, even though it's
        # not supported in Keras.
        bidaf_question_model.get_output_mask_shape_for = self.bidaf_question_model_mask_shape
        embedded_options = TimeDistributedWithMask(bidaf_question_model, keep_dims=True)(options_input)

        # Then we compare the weighted passage to each of the encoded options, and get a
        # distribution over answer options.  We'll use an encoder to get a single vector for the
        # passage and for each answer option, then do an "attention" to get a distribution over
        # answer options.  We can think of doing other similarity computations (e.g., a
        # decomposable attention) later.
        passage_encoder = self._get_encoder(name="similarity", fallback_behavior="use default params")
        option_encoder = EncoderWrapper(passage_encoder)
        encoded_passage = passage_encoder(weighted_passage)
        encoded_options = option_encoder(embedded_options)
        attention_layer = Attention(**deepcopy(self.similarity_function_params).as_dict())
        option_scores = attention_layer([encoded_passage, encoded_options])

        return DeepQaModel(inputs=[question_input, passage_input, options_input],
                           outputs=option_scores)

    @staticmethod
    def bidaf_question_model_mask_shape(input_shape):
        return input_shape[:-1]

    @overrides
    def _instance_type(self):  # pylint: disable=no-self-use
        return McQuestionPassageInstance

    @overrides
    def _get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = self._bidaf_model._get_padding_lengths()
        padding_lengths['num_options'] = self.num_options
        padding_lengths['num_option_words'] = self.num_option_words
        return padding_lengths

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        self._bidaf_model._set_padding_lengths(padding_lengths)
        self.num_options = padding_lengths['num_options']
        self.num_option_words = padding_lengths['num_option_words']

    @overrides
    def _set_padding_lengths_from_model(self):
        self._bidaf_model._set_padding_lengths_from_model()
        options_input_shape = self.model.get_input_shape_at(0)[2]
        self.num_options = options_input_shape[1]
        self.num_option_words = options_input_shape[2]

    @classmethod
    def _get_custom_objects(cls):
        custom_objects = BidirectionalAttentionFlow._get_custom_objects()
        custom_objects['Attention'] = Attention
        custom_objects['EncoderWrapper'] = EncoderWrapper
        custom_objects['Envelope'] = Envelope
        custom_objects['Multiply'] = Multiply
        custom_objects['TimeDistributedWithMask'] = TimeDistributedWithMask
        # Above, in `_build_model`, we do a total hack to make the partial BiDAF model compatible
        # with TimeDistributedWithMask.  We need a similar hack here, because we need Keras to have
        # this hacked `compute_output_mask_for` method when it loads the model from a config.  This
        # is really brittle...
        class DeepQaModelWithOutputMaskFunction(DeepQaModel):
            def get_output_mask_shape_for(self, input_shape):  # pylint: disable=no-self-use
                return input_shape[:-1]
        custom_objects['DeepQaModel'] = DeepQaModelWithOutputMaskFunction
        return custom_objects
