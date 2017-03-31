from keras import backend as K

from .word_alignment import WordAlignmentEntailment
from ...tensors.backend import switch


class MultipleChoiceTupleEntailment(WordAlignmentEntailment):
    '''A kind of decomposable attention where the premise (or background) is in
    the form of SVO triples, and entailment is computed by finding the answer
    in a multiple choice setting that aligns best with the tuples that align
    with the question. This happens in two steps:

    (1) We use the _align function from WordAlignmentEntailment to find the
        premise tuples whose SV, or VO pairs align best with the question.

    (2) We then use the _align function again to find the answer that aligns
        best with the unaligned part of the tuples, weighed by how much they
        partially align with the question in step 1.

    TODO(pradeep): Also match S with question, VO with answer, O with question
    and SV with answer.

    '''
    def __init__(self, **kwargs):
        self.tuple_size = None
        self.num_tuples = None
        self.num_options = None
        self.question_length = None
        super(MultipleChoiceTupleEntailment, self).__init__(**kwargs)

    def build(self, input_shape):
        #NOTE: This layer currently has no trainable parameters.
        super(MultipleChoiceTupleEntailment, self).build(input_shape)
        # knowledge_shape: (batch_size, num_tuples, tuple_size, embed_dim)
        # question_shape: (batch_size, question_length, embed_dim)
        # answer_shape: (batch_size, num_options, embed_dim)
        knowledge_shape, question_shape, answer_shape = input_shape
        self.tuple_size = knowledge_shape[2]
        if self.tuple_size != 3:
            raise NotImplementedError("Only SVO tuples are currently supported.")
        self.num_tuples = knowledge_shape[1]
        self.question_length = question_shape[1]
        self.num_options = answer_shape[1]
        self.input_dim = knowledge_shape[-1]

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.num_options)

    def compute_mask(self, x, mask=None):
        # pylint: disable=unused-argument
        return None

    def call(self, x, mask=None):
        # We assume the tuples are SVO and each slot is represented as vector.
        # Moreover, we assume each answer option is encoded as a single vector.
        # knowledge_embedding: (batch_size, num_tuples, tuple_size, embed_dim)
        # question_embedding: (batch_size, question_length, embed_dim)
        # answer_embedding: (batch_size, num_options, embed_dim)
        knowledge_embedding, question_embedding, answer_embedding = x
        if mask is None:
            knowledge_mask = question_mask = answer_mask = None
        else:
            knowledge_mask, question_mask, answer_mask = mask
        if knowledge_mask is None:
            sv_knowledge_mask = vo_knowledge_mask = subj_knowledge_mask = obj_knowledge_mask = None
        else:
            # Take out the relevant parts for each part of the tuple and reshape SV and VO masks using
            # batch_flatten.
            # (batch_size, num_tuples*2)
            sv_knowledge_mask = K.batch_flatten(knowledge_mask[:, :, :2])
            # (batch_size, num_tuples*2)
            vo_knowledge_mask = K.batch_flatten(knowledge_mask[:, :, 1:])
            # (batch_size, num_tuples)
            subj_knowledge_mask = knowledge_mask[:, :, 0]
            # (batch_size, num_tuples)
            obj_knowledge_mask = knowledge_mask[:, :, 2]
        batch_size = K.shape(knowledge_embedding)[0]
        sv_knowledge = K.reshape(knowledge_embedding[:, :, :2, :],
                                 (batch_size, self.num_tuples*2, self.input_dim))
        vo_knowledge = K.reshape(knowledge_embedding[:, :, 1:, :],
                                 (batch_size, self.num_tuples*2, self.input_dim))
        # (batch_size, num_tuples, embed_dim)
        subj_knowledge = knowledge_embedding[:, :, 0, :]
        # (batch_size, num_tuples, embed_dim)
        obj_knowledge = knowledge_embedding[:, :, 2, :]

        ## Step A1: Align SV with question.
        # Source is question, target is SV knowledge
        # (batch_size, question_length, num_tuples*2)
        sv_question_knowledge_alignment = self._align(question_embedding, sv_knowledge, question_mask,
                                                      sv_knowledge_mask, normalize_alignment=False)
        # Sum probabilities over S and V slots. This is still a valid probability distribution.
        # (batch_size, question_length, num_tuples)
        sv_question_tuple_weights = K.sum(K.reshape(sv_question_knowledge_alignment,
                                                    (batch_size, self.question_length, self.num_tuples, 2)),
                                          axis=-1)
        # Average over question length. This is essentially the weights of tuples based on how well their
        # S and V slots align to any word in the question.
        # Insight: This is essentially \sum_{i} p_align(tuple | q_word_i) * p_imp(q_word_i), where q_word_i is
        # the ith word in the question, p_align is the alignment weight and p_imp is the importance of the
        # question word, and p_imp is uniform.
        # (batch_size, num_tuples)
        sv_tuple_weights = K.mean(sv_question_tuple_weights, axis=1)

        ## Step A2: Align answer with Obj.
        # Source is obj knowledge, target is answer
        # (batch_size, num_tuples, num_options)
        obj_knowledge_answer_alignment = self._align(obj_knowledge, answer_embedding, obj_knowledge_mask,
                                                     answer_mask, normalize_alignment=False)
        # (batch_size, num_tuples, num_options)
        tiled_sv_tuple_weights = K.dot(K.expand_dims(sv_tuple_weights), K.ones((1, self.num_options)))
        # Now we compute a weighted average over the tuples dimension, with the weights coming from how well
        # the tuples align with the question.
        # (batch_size, num_options)
        obj_answer_weights = K.sum(tiled_sv_tuple_weights * obj_knowledge_answer_alignment, axis=1)

        # Following steps are similar to what we did so far. Just substitute VO for SV and S for O.
        ## Step B1: Align VO with question
        vo_question_knowledge_alignment = self._align(question_embedding, vo_knowledge, question_mask,
                                                      vo_knowledge_mask, normalize_alignment=False)
        vo_question_tuple_weights = K.sum(K.reshape(vo_question_knowledge_alignment,
                                                    (batch_size, self.question_length, self.num_tuples, 2)),
                                          axis=-1)
        vo_tuple_weights = K.mean(vo_question_tuple_weights, axis=1)

        ## Step B2: Align answer with Subj
        subj_knowledge_answer_alignment = self._align(subj_knowledge, answer_embedding, subj_knowledge_mask,
                                                      answer_mask, normalize_alignment=False)
        tiled_vo_tuple_weights = K.dot(K.expand_dims(vo_tuple_weights), K.ones((1, self.num_options)))
        subj_answer_weights = K.sum(tiled_vo_tuple_weights * subj_knowledge_answer_alignment, axis=1)

        # We now select the element wise max of obj_answer_weights and subj_answer_weights as our final weights.
        # (batch_size, num_options)
        max_answer_weights = switch(K.greater(obj_answer_weights, subj_answer_weights),
                                    obj_answer_weights, subj_answer_weights)
        # Renormalizing max weights.
        return K.softmax(max_answer_weights)
