from keras.layers import Dense

'''
Knowledge backed scorers take an encoded sentence (or logical form) representation
and encoded representations of background facts related to the sentence, and summarize
the background information as a weighted average of the representations of background
facts, conditioned on the encoded sentence. For example, KnowledgeBackedDense can be
used as the first layer in an MLP to make a memory network.
'''
class KnowledgeBackedDense(Dense):
    """
    Take the input as a tensor i = (nb_samples, (knowledge_len+1), input_dim), 
    i[:, 0] is the encoding of the sentence, i[:, 1:] are the encodings of the 
    background facts. Attend to facts conditioned on the input sentence, and pass
    to MLP the concatenation (nb_samples, 2 * input_dim)
    """
    def __init__(self):
        raise NotImplementedError
    
