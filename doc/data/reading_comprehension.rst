Reading Comprehension Instances
===============================

These ``Instances`` are designed for the set of tasks known today as "reading comprehension", where
the input is a natural language question, a passage, and (optionally) some number of answer
options, and the output is either a (span begin index, span end index) decision over the passage,
or a classification decision over the answer options (if provided).


QuestionPassageInstances
------------------------

.. automodule:: deep_qa.data.instances.reading_comprehension.question_passage_instance
    :members:
    :undoc-members:
    :show-inheritance:

McQuestionPassageInstances
--------------------------

.. automodule:: deep_qa.data.instances.reading_comprehension.mc_question_passage_instance
    :members:
    :undoc-members:
    :show-inheritance:

CharacterSpanInstances
----------------------

.. automodule:: deep_qa.data.instances.reading_comprehension.character_span_instance
    :members:
    :undoc-members:
    :show-inheritance:
