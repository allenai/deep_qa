Entailment Models
=================

Entailment models take two sequences of text as input and make a classification
decision on the pair. Typically that decision represents whether one sentence
entails the other, but we'll use this family of models to represent any kind of
classification decision over pairs of text.

Inputs: Two text sequences

Output: Some classification decision (typically "entails/not entails", "entails/neutral/contradicts", or similar)


DecomposableAttention
---------------------

.. automodule:: deep_qa.models.entailment.decomposable_attention
    :members:
    :undoc-members:
    :show-inheritance:
