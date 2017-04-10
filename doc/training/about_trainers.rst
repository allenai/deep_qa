About Trainers
==============

A :class:`~deep_qa.training.trainer.Trainer` is the core interface to the DeepQA code.  Trainers
specify data, a model, and a way to train the model with the data. This module groups all of the
common code related to these things, making only minimal assumptions about what kind of data you're
using or what the structure of your model is. Really, a ``Trainer`` is just a nicer interface to a
Keras ``Model``, we just call it something else to not create too much naming confusion, and
because the Trainer class provides a lot of functionality around training the model that a Keras
``Model`` doesn't.

On top of ``Trainer``, which is a nicer interface to a Keras ``Model``, this module provides a
``TextTrainer``, which adds a lot of functionality for building Keras ``Models`` that work with
text.  We provide APIs around word embeddings, sentence encoding, reading and padding datasets, and
similar things.  All of the concrete models that we have so far in DeepQA inherit from
``TextTrainer``, so understanding how to use this class is pretty important to understanding
DeepQA.

We also deal with the notion of pre-training in this module. A Pretrainer is a Trainer that depends
on another Trainer, building its model using pieces of the enclosed Trainer, so that training the
Pretrainer updates the weights in the enclosed Trainer object.
