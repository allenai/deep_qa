Trainer
=======

.. module:: deep_qa.training.trainer

.. autoclass:: Trainer

Public methods
~~~~~~~~~~~~~~

.. automethod:: Trainer.can_train
.. automethod:: Trainer.load_model
.. automethod:: Trainer.prepare_data
.. automethod:: Trainer.score_dataset
.. automethod:: Trainer.score_instance
.. automethod:: Trainer.train

Abstract methods
~~~~~~~~~~~~~~~~

.. automethod:: Trainer._build_model
.. automethod:: Trainer._load_dataset_from_files
.. automethod:: Trainer._prepare_data
.. automethod:: Trainer._prepare_instance
.. automethod:: Trainer._set_params_from_model

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
