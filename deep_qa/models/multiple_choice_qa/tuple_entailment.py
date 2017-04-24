from typing import Dict, List

from overrides import overrides

from keras.layers import Input

from ...common.params import Params
from ...data.instances.multiple_choice_qa import QuestionAnswerInstance
from ...data.instances.text_classification import TupleInstance
from ...data.instances.wrappers import read_background_from_file
from ...layers.entailment_models import MultipleChoiceTupleEntailment
from ...layers.wrappers import EncoderWrapper

from ...training.text_trainer import TextTrainer
from ...training.models import DeepQaModel


class MultipleChoiceTupleEntailmentModel(TextTrainer):
    '''
    This solver reads question and answer text, and background information in the form of tuples, and
    uses a tuple alignment entailment model to obtain a probability distribution over the answer options
    based on how well they align with the background tuples, given the question text.
    '''
    def __init__(self, params: Params):
        self.entailment_model_params = params.pop('entailment_model', {})
        self.max_answer_length = params.pop('max_answer_length', None)
        self.max_knowledge_length = params.pop('max_knowledge_length', None)
        self.num_slots = params.pop('num_tuple_slots', 3)
        self.num_options = params.pop('num_answer_options', 4)
        super(MultipleChoiceTupleEntailmentModel, self).__init__(params)

        self.name = 'MultipleChoiceTupleEntailmentModel'
        self.entailment_choices = ['multiple_choice_tuple_attention']

    @overrides
    def _instance_type(self):
        return QuestionAnswerInstance

    @staticmethod
    def _background_instance_type():
        return TupleInstance

    @overrides
    def load_dataset_from_files(self, files: List[str]):
        dataset = super(MultipleChoiceTupleEntailmentModel, self).load_dataset_from_files(files)
        return read_background_from_file(dataset, files[1], self._background_instance_type())

    @classmethod
    @overrides
    def _get_custom_objects(cls):
        custom_objects = super(MultipleChoiceTupleEntailmentModel, cls)._get_custom_objects()
        custom_objects['MultipleChoiceTupleEntailment'] = MultipleChoiceTupleEntailment
        return custom_objects

    @overrides
    def _get_padding_lengths(self) -> Dict[str, int]:
        return {
                'num_sentence_words': self.num_sentence_words,
                'answer_length': self.max_answer_length,
                'background_sentences': self.max_knowledge_length,
                'num_options': self.num_options,
                'num_slots': self.num_slots
                }

    @overrides
    def _set_padding_lengths(self, padding_lengths: Dict[str, int]):
        if self.num_sentence_words is None:
            self.num_sentence_words = padding_lengths['num_sentence_words']
        if self.max_answer_length is None:
            self.max_answer_length = padding_lengths['answer_length']
        if self.max_knowledge_length is None:
            self.max_knowledge_length = padding_lengths['background_sentences']
        if self.num_options is None:
            self.num_options = padding_lengths['num_options']
        if self.num_slots is None:
            self.num_slots = padding_lengths['num_slots']

    @overrides
    def _set_padding_lengths_from_model(self):
        # TODO(matt): actually implement this (it's required for loading a saved model)
        pass

    @overrides
    def _build_model(self):
        question_input = Input((self.num_sentence_words,), dtype='int32', name='question_input')
        answer_input = Input((self.num_options, self.max_answer_length), dtype='int32', name='answer_input')
        knowledge_input = Input((self.max_knowledge_length, self.num_slots, self.num_sentence_words),
                                dtype='int32', name='knowledge_input')
        question_embedding = self._embed_input(question_input)
        answer_embedding = self._embed_input(answer_input)
        knowledge_embedding = self._embed_input(knowledge_input)

        # We need to encode just the answer and knowledge, not the question.
        # We'll use the same encoder for both, but we just need to wrap the encoder one more time for knowledge.
        sentence_encoder = self._get_encoder()
        answer_encoder = EncoderWrapper(sentence_encoder, name='answer_encoder')
        knowledge_encoder = EncoderWrapper(EncoderWrapper(sentence_encoder), name='knowledge_encoder')

        encoded_answers = answer_encoder(answer_embedding)
        encoded_knowledge = knowledge_encoder(knowledge_embedding)

        entailment_layer = MultipleChoiceTupleEntailment(**self.entailment_model_params)
        entailment_output = entailment_layer([encoded_knowledge, question_embedding, encoded_answers])

        return DeepQaModel(input=[question_input, knowledge_input, answer_input], output=entailment_output)
