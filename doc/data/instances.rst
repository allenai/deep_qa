Data Instances
==============

An Instance is a single training or testing example for a Keras model. The base
classes for working with Instances are found in instance.py. There are two
subclasses: (1) TextInstance, which is a raw instance that contains actual
strings, and can be used to determine a vocabulary for a model, or read directly
from a file; and (2) IndexedInstance, which has had its raw strings converted to
word (or character) indices, and can be padded to a consistent length and
converted to numpy arrays for use with Keras.

There are a lot of different concrete Instance objects you can use. Some
examples:

* A TrueFalseInstance, that contains a single sentence with a true/false label.
  The numpy array for this instance is just a single word index sequence.
* A MultipleTrueFalseInstance, which contains several TrueFalseInstances, only
  one of which is true. The numpy array here has shape (num_options,
  sentence_length), and the label is a one-hot vector of length num_options.
* A BackgroundInstance, which wraps another Instance type with a set of
  background sentences, adding an additional input of size (knowledge_length,
  sentence_length).
* A LogicalFormInstance, which is a TrueFalseInstance where the "sentence" is
  actually a tree-structured logical form (hmm, maybe we should call this a
  TreeInstance instead... TODO(matt).). In addition to the numpy array
  containing the word index sequence, there's another array containing shift /
  reduce operations so that you can construct a tree-structured network using a
  sequence, like in the SPINN paper by Sam Bowman and others (see the
  TreeCompositionLSTM encoder for a way to actually use this in a model).

A lot of the magic of how the DeepQA library works happens here, in the concrete
Instance classes in this module. Most of the code can be totally agnostic to how
exactly the input is structured, because the conversion to numpy arrays happens
here, not in the Trainer or Solver classes, with only the specific
_build_model() methods needing to know about the format of their input and
output.

babi_instance
-------------

.. automodule:: deep_qa.data.instances.babi_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

background_instance
-------------------

.. automodule:: deep_qa.data.instances.background_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

character_span_instance
-----------------------

.. automodule:: deep_qa.data.instances.character_span_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

instance
--------

.. automodule:: deep_qa.data.instances.instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

labeled_background_instance
---------------------------

.. automodule:: deep_qa.data.instances.labeled_background_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

logical_form_instance
---------------------

.. automodule:: deep_qa.data.instances.logical_form_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

mc_question_answer_instance
---------------------------

.. automodule:: deep_qa.data.instances.mc_question_answer_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

multiple_true_false_instance
----------------------------

.. automodule:: deep_qa.data.instances.multiple_true_false_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

question_answer_instance
------------------------

.. automodule:: deep_qa.data.instances.question_answer_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

question_passage_instance
-------------------------

.. automodule:: deep_qa.data.instances.question_passage_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

sentence_pair_instance
----------------------

.. automodule:: deep_qa.data.instances.sentence_pair_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

snli_instance
-------------

.. automodule:: deep_qa.data.instances.snli_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

true_false_instance
-------------------

.. automodule:: deep_qa.data.instances.true_false_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

tuple_inference_instance
---------------------------

.. automodule:: deep_qa.data.instances.tuple_inference_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:

tuple_instance
--------------

.. automodule:: deep_qa.data.instances.tuple_instance
    :members:
    :noindex:
    :undoc-members:
    :show-inheritance:
