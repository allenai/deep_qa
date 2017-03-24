About Models
============

In this module we define a number of concrete models. The models are grouped by
task, where each task has a roughly coherent input/output specification. See the
README in each submodule for a description of the task models in that submodule
are designed to solve.

You should think of these models as more of "model families" than actual models,
though, as there are typically options left unspecified in the models
themselves. For example, models in this module might have a layer that encodes
word sequences into vectors; they just call a method on TextTrainer to get an
encoder, and the decision for which actual encoder is used (an LSTM, a CNN, or
something else) happens in the parameters passed to TextTrainer. If you really
want to, you can hard-code specific decisions for these things, but most models
we have here use the TextTrainer API to abstract away these decisions, giving
implementations of a class of similar models, instead of a single model.

We also define a few general Pretrainers in a submodule here. The Pretrainers in
this top-level submodule are suitable to pre-train a large class of models
(e.g., any model that encodes sentences), while more task-specific Pretrainers
are found in that task's submodule.

Below, we describe a few popular models that we've implemented and include our
output when training.

Attention Sum Reader
--------------------

The `Attention Sum Reader
<https://www.semanticscholar.org/paper/Text-Understanding-with-the-Attention-Sum-Reader-Kadlec-Schmid/1023b20d226bd0af9fdf0fd1847accefbfa5ec84>`_
Network is implemented in
:mod:`~deep_qa.models.reading_comprehension.attention_sum_reader`.

.. container:: toggle

    .. container:: header

        **Press to show/hide train logs**

    Train Logs::

        Using Theano backend.
        Using gpu device 0: Tesla K80 (CNMeM is disabled, cuDNN 5105)
        /home/nelsonl/miniconda3/envs/deep_qa/lib/python3.5/site-packages/theano/sandbox/cuda/__init__.py:600: UserWarning: Your cuDNN version is more recent than the one Theano officially supports. If you see any problems, try updating Theano or downgrading cuDNN to version 5.
          warnings.warn(warn)
        2017-01-26 23:52:54,082 - INFO - deep_qa.common.checks - Keras version: 1.2.0
        2017-01-26 23:52:54,082 - INFO - deep_qa.common.checks - Theano version: 0.8.2
        2017-01-26 23:52:54,269 - INFO - __main__ - Training model
        2017-01-26 23:52:54,270 - INFO - deep_qa.training.trainer - Running training (TextTrainer)
        2017-01-26 23:52:54,270 - INFO - deep_qa.training.trainer - Getting training data
        2017-01-26 23:52:58,914 - INFO - deep_qa.data.dataset - Finished reading dataset; label counts: [(0, 42399), (1, 44896), (2, 23832), (3, 11274), (4, 585)]
        2017-01-26 23:58:07,539 - INFO - deep_qa.training.text_trainer - Indexing dataset
        2017-01-27 00:03:28,722 - INFO - deep_qa.training.text_trainer - Padding dataset to lengths {'num_option_words': None, 'num_question_words': None, 'wod_sequence_length': None, 'num_options': None, 'num_passage_words': None}
        2017-01-27 00:03:28,722 - INFO - deep_qa.data.dataset - Getting max lengths from instances
        2017-01-27 00:03:29,714 - INFO - deep_qa.data.dataset - Instance max lengths: {'num_option_words': 68, 'num_question_words': 121, 'num_options': 5, 'nm_passage_words': 3090}
        2017-01-27 00:03:29,714 - INFO - deep_qa.data.dataset - Now actually padding instances to length: {'num_option_words': 68, 'num_question_words': 121, num_options': 5, 'num_passage_words': 3090}
        2017-01-27 00:05:40,054 - INFO - deep_qa.training.trainer - Getting validation data
        2017-01-27 00:05:40,347 - INFO - deep_qa.data.dataset - Finished reading dataset; label counts: [(0, 3522), (1, 3429), (2, 1835), (3, 784), (4, 430)]
        2017-01-27 00:05:40,348 - INFO - deep_qa.training.text_trainer - Indexing dataset
        2017-01-27 00:06:02,773 - INFO - deep_qa.training.text_trainer - Padding dataset to lengths {'num_option_words': 68, 'num_question_words': 121, 'word_sequence_length': None, 'num_options': 5, 'num_passage_words': 3090}
        2017-01-27 00:06:02,774 - INFO - deep_qa.data.dataset - Getting max lengths from instances
        2017-01-27 00:06:02,851 - INFO - deep_qa.data.dataset - Instance max lengths: {'num_option_words': 8, 'num_question_words': 95, 'num_options': 5, 'num_passage_words': 2186}
        2017-01-27 00:06:02,851 - INFO - deep_qa.data.dataset - Now actually padding instances to length: {'num_option_words': 68, 'num_question_words': 121, 'num_options': 5, 'num_passage_words': 3090}
        2017-01-27 00:06:13,387 - INFO - deep_qa.training.trainer - Building the model
        ____________________________________________________________________________________________________
        Layer (type)                     Output Shape          Param #     Connected to
        ====================================================================================================
        document_input (InputLayer)      (None, 3090)          0
        ____________________________________________________________________________________________________
        question_input (InputLayer)      (None, 121)           0
        ____________________________________________________________________________________________________
        word_embedding (TimeDistributedE multiple              80112384    question_input[0][0]
                                                                           document_input[0][0]
        ____________________________________________________________________________________________________
        bidirectional_1 (Bidirectional)  (None, 768)           1476864     word_embedding[0][0]
        ____________________________________________________________________________________________________
        bidirectional_2 (Bidirectional)  (None, 3090, 768)     1476864     word_embedding[1][0]
        ____________________________________________________________________________________________________
        question_document_softmax (Atten (None, 3090)          0           bidirectional_1[0][0]
                                                                           bidirectional_2[0][0]
        ____________________________________________________________________________________________________
        options_input (InputLayer)       (None, 5, 68)         0
        ____________________________________________________________________________________________________
        options_probability_sum (OptionA (None, 5)             0           document_input[0][0]
                                                                           question_document_softmax[0][0]
                                                                           options_input[0][0]
        ____________________________________________________________________________________________________
        l1normalize_1 (L1Normalize)      (None, 5)             0           options_probability_sum[0][0]
        ====================================================================================================
        Total params: 83,066,112
        Trainable params: 83,066,112
        Non-trainable params: 0
        ____________________________________________________________________________________________________
        Train on 127786 samples, validate on 10000 samples
        Epoch 1/5
        127786/127786 [==============================] - 34850s - loss: 1.0131 - acc: 0.5290 - val_loss: 0.9776 - val_acc: 0.5624
        Epoch 2/5
        127786/127786 [==============================] - 34828s - loss: 0.6713 - acc: 0.7267 - val_loss: 1.0838 - val_acc: 0.5514
        Epoch 3/5
        127786/127786 [==============================] - 34835s - loss: 0.2720 - acc: 0.8996 - val_loss: 1.4446 - val_acc: 0.5335
