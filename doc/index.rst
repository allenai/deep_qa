.. deep_qa documentation master file, created by
   sphinx-quickstart on Wed Jan 25 11:35:07 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Home
====

This repository contains code for training deep learning systems to do question
answering tasks. Our primary focus is on Aristo's science questions, though we
can run various models on several popular datasets.

This code is a mix of scala (for data processing / pipeline management) and
python (for actually training and executing deep models with Keras / Theano /
TensorFlow).

I think there are two main contributions that this library makes:

1. We provide a nice interface to training, validating, and debugging Keras
models. Instead of writing code to run an experiment, you just specify a JSON
file that describes your experiment, and our library will run it for you
(assuming you only want to use the components that we've implemented).

2. We've implemented several variants of memory networks, neural network
architectures that attempt a kind of reasoning over background knowledge, that
are not trivial to implement. We've done this in a way that is configurable and
extendable, such that you can easily run experiments with several different
memory network variants just by changing some parameters in a configuration
file, or implement a new idea just by writing a new component.

This library has several main components:

* A training module, which has a bunch of helper code for training Keras models
  of various kinds.
* A solvers module, specifying particular Keras models for question answering
  (in Aristo, we call a question answering system a "solver", which is where the
  name comes from).
* A layers module, which contains code for custom Keras Layers that we have
  written.
* A data module, containing code for reading in data from files and converting
  it into numpy arrays suitable for use with Keras.
* A common module, which has a few random things dealing with reading parameters
  and a few other things.

.. toctree::
   :hidden:

   self

.. toctree::
   :caption: Layers
   :hidden:

   layers/about_layers
   layers/core_layers
   layers/backend
   layers/encoders
   layers/entailment_models
   layers/attention

.. toctree::
   :caption: Models
   :hidden:

   models/about_models
   models/entailment
   models/memory_networks
   models/memory_networks_pretrainers
   models/multiple_choice_qa
   models/reading_comprehension
   models/text_classification
   models/text_pretrainers

.. toctree::
   :caption: Data
   :hidden:

   data/about_data
   data/general_data_utils
   data/instances
   data/tokenizers

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

.. toctree::
   :caption: Trainer Utils
   :hidden:

   training/about_training
   training/core_training
   training/pretraining

.. toctree::
   :hidden:
   :maxdepth: 4
   :caption: Other

   api_doc/deep_qa
