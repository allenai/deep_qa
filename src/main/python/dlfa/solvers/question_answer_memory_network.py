from typing import Dict
from overrides import overrides

from keras.layers import TimeDistributed

from ..data.dataset import TextDataset
from ..data.text_instance import QuestionAnswerInstance
from .memory_network import MemoryNetworkSolver


class QuestionAnswerMemoryNetworkSolver(MemoryNetworkSolver):
    '''
    This is a MemoryNetworkSolver that is trained on QuestionAnswerInstances.

    The base MemoryNetworkSolver assumes we're dealing with TrueFalseInstances, so there is no
    separate question and answer text.  We need to add an additional input, and handle it specially
    at the end, with a very different final "entailment" model, where we're doing dot-product
    similarity with encoded answer options, or something similar.
    '''

    entailment_choices = ['question_answer_mlp']
    entailment_default = entailment_choices[0]
    def __init__(self, **kwargs):
        super(QuestionAnswerMemoryNetworkSolver, self).__init__(**kwargs)
        self.num_options = None
        self.max_answer_length = None

    @overrides
    def can_train(self) -> bool:
        """
        Where a MemoryNetworkSolver allows separate positive and negative training files, we only
        allow a single train file, so we need to override this method.

        The train file must be a valid question file, as determined by
        Dataset.can_be_converted_to_multiple_choice(), but we don't check that here.
        """
        has_train = self.train_file is not None and self.train_background is not None
        has_validation = self.validation_file is not None and self.validation_background is not None
        return has_train and has_validation

    @overrides
    def _instance_type(self):
        return QuestionAnswerInstance

    @overrides
    def _get_max_lengths(self) -> Dict[str, int]:
        return {
                'word_sequence_length': self.max_sentence_length,
                'answer_length': self.max_answer_length,
                'num_options': self.num_options,
                'background_sentences': self.max_knowledge_length,
                }

    @overrides
    def _set_max_lengths(self, max_lengths: Dict[str, int]):
        self.max_sentence_length = max_lengths['word_sequence_length']
        self.max_answer_length = max_lengths['answer_length']
        self.num_options = max_lengths['num_options']
        self.max_knowledge_length = max_lengths['background_sentences']

    @overrides
    def _get_entailment_output(self, combined_input):
        answer_input_layer, answer_embedding = self._get_embedded_sentence_input(
                input_shape=(self.num_options, self.max_answer_length), name_prefix="answer")
        answer_encoder = TimeDistributed(self._get_new_encoder(), name="answer_encoder")
        encoded_answers = answer_encoder(answer_embedding)
        return ([answer_input_layer],
                self.entailment_model.classify(combined_input, encoded_answers, self.embedding_size))

    @overrides
    def _get_validation_data(self):
        dataset = TextDataset.read_from_file(self.validation_file,
                                             self._instance_type(),
                                             tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset, self.validation_background)
        self.validation_dataset = background_dataset
        return self.prep_labeled_data(background_dataset, for_train=False, shuffle=True)

    @overrides
    def _get_test_data(self):
        dataset = TextDataset.read_from_file(self.test_file,
                                             self._instance_type(),
                                             tokenizer=self.tokenizer)
        background_dataset = TextDataset.read_background_from_file(dataset, self.test_background)
        return self.prep_labeled_data(background_dataset, for_train=False, shuffle=True)

    @overrides
    def evaluate(self, labels, test_input):
        """
        We need to override this method, because our test input is already grouped by question.
        """
        scores = self.model.evaluate(test_input, labels)
        return scores[1]  # NOTE: depends on metrics=['accuracy'] in self.model.compile()
