from typing import Dict
from overrides import overrides

from keras.layers import merge, Dense, Input

from .memory_network import MemoryNetwork
from ...data.instances.multiple_choice_qa.babi_instance import BabiInstance
from ...common.params import Params
from ...training.models import DeepQaModel
from ...layers import VectorMatrixMerge


class SoftmaxMemoryNetwork(MemoryNetwork):
    '''
    A SoftmaxMemoryNetwork is a memory network that ends with a final softmax over a fixed
    set of answer options.  These answer options must not change during training or testing, or
    this model will break.  If you want to run experiments on the bAbI dataset, this is the model
    you should use.

    I think this kind of model is totally unrealistic for answering real questions, but it's what
    the original memory networks did, so in an attempt to re-implement their models exactly, we
    have this version of the memory network.

    The behavior of this solver is similar to the QuestionAnswerMemoryNetwork, except the QAMN
    embeds the answer and has an explicit similarity comparison followed by a softmax over the
    similarities, instead of the softmax over all possible answer options done here (which would be
    over an infinite set in a real science exam).
    '''
    has_sigmoid_entailment = False
    has_multiple_backgrounds = False

    def __init__(self, params: Params):
        super(SoftmaxMemoryNetwork, self).__init__(params)
        self.name = "SoftmaxMemoryNetwork"
        self.num_options = None
        self.knowledge_encoders = {}

    @overrides
    def _instance_type(self):
        return BabiInstance

    @overrides
    def _get_padding_lengths(self) -> Dict[str, int]:
        padding_lengths = super(SoftmaxMemoryNetwork, self)._get_padding_lengths()
        padding_lengths['num_options'] = self.num_options
        padding_lengths['answer_length'] = 1  # because BabiInstance inherits from QuestionAnswerInstance...
        return padding_lengths

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        super(SoftmaxMemoryNetwork, self)._set_padding_lengths(padding_lengths)
        if self.num_options is None:
            self.num_options = padding_lengths['num_options']

    @overrides
    def _build_model(self):
        # TODO(matt): I'm starting out with overriding this method, instead of trying to generalize
        # the base class.  Hopefully we'll be able to merge these, but I wanted to start out with
        # something simple.

        # Steps 1 and 2: Convert inputs to sequences of word vectors, for both the question
        # inputs and the knowledge inputs.
        question_input = Input(shape=self._get_question_shape(), dtype='int32', name="sentence_input")
        knowledge_input = Input(shape=self._get_background_shape(), dtype='int32', name="background_input")
        question_embedding = self._embed_input(question_input, embedding_name="embedding_B")

        # Step 3: Encode the two embedded inputs using the sentence encoder.
        question_encoder = self._get_encoder()
        encoded_question = question_encoder(question_embedding)  # (samples, word_dim)


        # Step 4: Merge the two encoded representations and pass into the knowledge backed scorer.
        # At each step in the following loop, we take the question encoding, or the output of
        # the previous memory layer, merge it with the knowledge encoding and pass it to the
        # current memory layer.
        current_memory = encoded_question

        knowledge_combiner = self._get_knowledge_combiner(0)
        knowledge_axis = self._get_knowledge_axis()
        for i in range(self.num_memory_layers):
            knowledge_encoder = self._get_knowledge_encoder(question_encoder, name='knowledge_encoder_A' + str(i))
            knowledge_embedding = self._embed_input(knowledge_input, embedding_name="embedding_A" + str(i))
            encoded_knowledge = knowledge_encoder(knowledge_embedding)

            # We do this concatenation so that the knowledge selector can be TimeDistributed
            # correctly.
            merged_encoded_rep = VectorMatrixMerge(
                    concat_axis=knowledge_axis,
                    name='concat_memory_layer_%d' % i)([encoded_question, current_memory, encoded_knowledge])

            knowledge_selector = self._get_knowledge_selector(i)
            attention_weights = knowledge_selector(merged_encoded_rep)

            output_knowledge_encoder = self._get_knowledge_encoder(question_encoder,
                                                                   name='knowledge_encoder_A' + str(i+1))
            output_embedding = self._embed_input(knowledge_input, embedding_name="embedding_A" + str(i+1))
            encoded_output_knowledge = output_knowledge_encoder(output_embedding)

            # Again, this concatenation is done so that we can TimeDistribute the knowledge
            # combiner.  TimeDistributed only allows a single input, so we need to merge them.
            combined_background_with_attention = VectorMatrixMerge(
                    concat_axis=knowledge_axis + 1,
                    propagate_mask=False,  # the attention will have zeros, so we don't need a mask
                    name='concat_attention_%d' % i)([attention_weights, encoded_output_knowledge])
            attended_knowledge = knowledge_combiner(combined_background_with_attention)

            # This one is just a regular merge, because all of these are vectors.  The
            # concatenation is done, you guessed it, so that the memory_updater can be
            # TimeDistributed.
            updater_input = merge([encoded_question, current_memory, attended_knowledge],
                                  mode='concat',
                                  concat_axis=knowledge_axis,
                                  name='concat_current_memory_with_background_%d' % i)
            memory_updater = self._get_memory_updater(i)
            current_memory = memory_updater(updater_input)

        final_softmax = Dense(units=self.num_options, activation='softmax', name='final_softmax')
        output = final_softmax(current_memory)

        input_layers = [question_input, knowledge_input]
        return DeepQaModel(input=input_layers, output=output)
