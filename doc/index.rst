.. deep_qa documentation master file, created by
   sphinx-quickstart on Wed Jan 25 11:35:07 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

DeepQA is a library built on top of Keras to make NLP easier.  There are four main benefits to
this library:

#. It is hard to get NLP right in Keras.  There are a lot of issues around padding sequences and
   masking that are not handled well in the main Keras code, and we have well-tested code that
   does the right thing for, e.g., computing attentions over padded sequences, or distributing text
   encoders across several sentences or words.
#. We have implemented a base class, :class:`~deep_qa.training.text_trainer.TextTrainer`, that
   provides a nice, consistent API around building NLP models in Keras.  This API has functionality
   around processing data instances, embedding words and/or characters, easily getting various
   kinds of sentence encoders, and so on.
#. We provide a nice interface to training, validating, and debugging Keras models.  It is very
   easy to experiment with variants of a model family, just by changing some parameters in a JSON
   file.  For example, you can go from using fixed GloVe vectors to represent words, to fine-tuning
   those embeddings, to using a concatenation of word vectors and a character-level CNN to
   represent words, just by changing parameters in a JSON experiment file.  If your model is built
   using the ``TextTrainer`` API, all of this works transparently to the model class - the model
   just knows that it's getting some kind of word vector.
#. We have implemented a number of state-of-the-art models, particularly focused around question
   answering systems (though we've dabbled in models for other tasks, as well).  The actual model
   code for these systems are typically 50 lines or less.

This library has several main components:

* A ``training`` module, which has a bunch of helper code for training Keras models of various
  kinds.
* A ``models`` module, containing implementations of actual Keras models grouped around various
  prediction tasks.
* A ``layers`` module, which contains code for custom Keras Layers that we have written.
* A ``data`` module, containing code for reading in data from files and converting it into numpy
  arrays suitable for use with Keras.
* A ``common`` module, which has a few random things dealing with reading parameters and a few
  other things.

.. toctree::
   :hidden:

   self

.. toctree::
   :caption: Training
   :hidden:

   training/about_trainers
   training/trainer
   training/text_trainer
   training/misc

.. toctree::
   :caption: Data
   :hidden:

   data/about_data
   data/instances
   data/entailment
   data/multiple_choice_qa
   data/reading_comprehension
   data/sentence_selection
   data/sequence_tagging
   data/text_classification
   data/wrappers
   data/tokenizers
   data/general_data_utils

.. toctree::
   :caption: Models
   :hidden:

   models/about_models
   models/entailment
   models/memory_networks
   models/multiple_choice_qa
   models/sentence_selection
   models/reading_comprehension
   models/text_classification

.. toctree::
   :caption: Layers
   :hidden:

   layers/about_layers
   layers/core_layers
   layers/attention
   layers/backend
   layers/encoders
   layers/entailment_models
   layers/tuple_matchers
   layers/wrappers

.. toctree::
   :caption: Tensor Utils
   :hidden:

   tensors/about_tensors
   tensors/core_tensors
   tensors/similarity_functions

.. toctree::
   :caption: Common Utils
   :hidden:

   common/about_common
   common/checks
   common/params
