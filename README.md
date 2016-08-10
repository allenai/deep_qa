# Deep Learning for Aristo

This repository contains code for training a deep learning system to answer science questions.  The
architecture of this system is based on two main assumptions:

1. We do not have enough question-answer pairs to train a deep model on these questions, so we
   must instead train to predict whether a sentence is true or false, and convert question-answer
pairs to sentences at test time.
2. We assume our science questions are complex enough that we want the system to have direct
   access to background knowledge over which it can reason while deciding whether a sentence is
true or false.

This leads to six main components of this architecture, which are described in more detail in a
[design
document](https://docs.google.com/document/d/1sLJ-W-ylpEqwGyZ8WdM5O_n3aso5ONfKSnVVlAsZ1hk/edit#):

1. A *question interpreter*, which converts question-answer pairs into declarative sentences.
2. A *training data selector*, which selects (or constructs) declarative sentences from a large
   corpus for use in training the deep learning model.
3. A *sentence corruptor*, which changes the positive examples from step 2 in some way such that
   they are false, and can be used as negative examples to train the deep learning model (either
as explicit negatives, or with a ranking loss).
4. A *background knowledge base and/or corpus*.  I guess this is an input, not a step in the
   architecture, at this point.
5. *Background knowledge search*, which, given a sentence, selects relevant parts of the
   background KB / corpus that will be input to the deep neural network.
6. A *neural network* that scores sentences as true or false.  The simplest baselines for this
   make no use of background knowledge, but we also are implementing memory networks that can
reason over the background knowledge to answer the question.
