TextTrainer
===========

.. module:: deep_qa.training.text_trainer

.. autoclass:: TextTrainer

Utility methods
~~~~~~~~~~~~~~~

These methods are intended for use by subclasses, mostly in your ``_build_model`` implementation.

.. automethod:: TextTrainer._get_sentence_shape
.. automethod:: TextTrainer._embed_input
.. automethod:: TextTrainer._get_encoder
.. automethod:: TextTrainer._get_seq2seq_encoder
.. automethod:: TextTrainer._set_text_lengths_from_model_input

Abstract methods
~~~~~~~~~~~~~~~~

.. automethod:: TextTrainer._instance_type
.. automethod:: TextTrainer._set_max_lengths_from_model

Semi-abstract methods
~~~~~~~~~~~~~~~~~~~~~

You'll likely need to override these methods, if you have anything more complex than a single sentence
as input.

.. automethod:: TextTrainer._get_max_lengths
.. automethod:: TextTrainer._set_max_lengths

Overridden ``Trainer`` methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You probably don't need to override these, except for probably ``_get_custom_objects``.  The rest
of them you shouldn't need to worry about at all, but we document them here for completeness.

.. automethod:: TextTrainer._get_custom_objects
.. automethod:: TextTrainer._load_auxiliary_files
.. automethod:: TextTrainer._load_dataset_from_files
.. automethod:: TextTrainer._overall_debug_output
.. automethod:: TextTrainer._prepare_data
.. automethod:: TextTrainer._prepare_instance
.. automethod:: TextTrainer._save_auxiliary_files
.. automethod:: TextTrainer._set_params_from_model
