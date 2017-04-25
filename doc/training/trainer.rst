Trainer
=======

.. module:: deep_qa.training.trainer

.. autoclass:: Trainer

Public methods
~~~~~~~~~~~~~~

.. automethod:: Trainer.can_train
.. automethod:: Trainer.load_model
.. automethod:: Trainer.load_data_arrays
.. automethod:: Trainer.score_dataset
.. automethod:: Trainer.train

Abstract methods
~~~~~~~~~~~~~~~~

If you're doing NLP, :class:`~deep_qa.training.text_trainer.TextTrainer` implements most of these,
so you shouldn't have to worry about them.  The only one it doesn't is ``_build_model`` (though it
adds some other abstract methods that you `might` have to worry about).

.. automethod:: Trainer.load_dataset_from_files
.. automethod:: Trainer.set_model_state_from_dataset
.. automethod:: Trainer.set_model_state_from_indexed_dataset
.. automethod:: Trainer.create_data_arrays
.. automethod:: Trainer._build_model
.. automethod:: Trainer._set_params_from_model
.. automethod:: Trainer._dataset_indexing_kwargs

Protected methods
~~~~~~~~~~~~~~~~~

.. automethod:: Trainer._get_callbacks
.. automethod:: Trainer._get_custom_objects
.. automethod:: Trainer._instance_debug_output
.. automethod:: Trainer._load_auxiliary_files
.. automethod:: Trainer._overall_debug_output
.. automethod:: Trainer._post_epoch_hook
.. automethod:: Trainer._pre_epoch_hook
.. automethod:: Trainer._save_auxiliary_files
.. automethod:: Trainer._uses_data_generators
