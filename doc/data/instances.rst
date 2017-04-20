Base Instances
==============

An :class:`~deep_qa.data.instances.instance.Instance` is a single training or testing example for a Keras model. The base classes for
working with ``Instances`` are found in instance.py. There are two subclasses: (1)
:class:`~deep_qa.data.instances.instance.TextInstance`, which is a raw instance that contains
actual strings, and can be used to determine a vocabulary for a model, or read directly from a
file; and (2) :class:`~deep_qa.data.instances.instance.IndexedInstance`, which has had its raw
strings converted to word (or character) indices, and can be padded to a consistent length and
converted to numpy arrays for use with Keras.

Concrete ``Instance`` classes are organized in the code by the task they are designed for (e.g.,
text classification, reading comprehension, sequence tagging, etc.).

A lot of the magic of how the DeepQA library works happens here, in the concrete Instance classes
in this module. Most of the code can be totally agnostic to how exactly the input is structured,
because the conversion to numpy arrays happens here, not in the Trainer or TextTrainer classes,
with only the specific ``_build_model()`` methods needing to know about the format of their input
and output (and even some of the details there are transparent to the model class).

.. automodule:: deep_qa.data.instances.instance
    :members:
    :undoc-members:
    :show-inheritance:
