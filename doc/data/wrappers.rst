Wrapper Instances
=================

These ``Instances`` wrap other ``Instances``, typically adding new information to the instance
object without changing the task that is performed.  For example, you might want to add some kind
of background knowledge to a question answering task, or you might want additional metadata about
a document for a sequence tagging task.  The idea is that these could be simple wrappers around
already-existing ``Instances`` that just add another data array that gets input to your model.

BackgroundInstances
-------------------

.. automodule:: deep_qa.data.instances.wrappers.background_instance
    :members:
    :undoc-members:
    :show-inheritance:
