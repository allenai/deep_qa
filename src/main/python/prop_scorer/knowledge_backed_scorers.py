from keras.layers import Dense

'''
Knowledge backed scorers take an encoded sentence (or logical form) representation
and encoded representations of background facts related to the sentence, and summarize
the background information as a weighted average of the representations of background
facts, conditioned on the encoded sentence. For example, KnowledgeBackedDense can be
used as the first layer in an MLP to make a memory network.
'''
class KnowledgeBackedDense(Dense):
    def __init__(self):
        raise NotImplementedError
    
