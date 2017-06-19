# DeepQA

DeepQA is organised into the following main sections:

-   `common`: Code for parameter parsing, logging and runtime checks.
-   `data`: Indexing, padding, tokenisation, stemming, embedding and general dataset manipulation
    happens here.
-   `layers`: The bulk of the library. Use these Layers to compose new models. Some of these Layers
    are very similar to what you might find in Keras, but altered slightly to support arbitrary
dimensions or correct masking.
-   `models`: Frameworks for different types of task. These generally all extend the TextTrainer
    class which provides training capabilities to a DeepQaModel. We have models for Sequence
Tagging, Entailment, Multiple Choice QA, Reading Comprehension and more. Take a look at the READMEs
under `model` for more details - each task typically has a README describing the task definition.
-   `tensors`: Convenience functions for writing the internals of Layers.  Will almost exclusively be
    used inside Layer implementations.
-   `training`: This module does the heavy lifting for training and optimisation. We also wrap the
    Keras Model class to give it some useful debugging functionality.

The `data` and `models` sections are, in turn, structured according to what task they are intended
for (e.g., text classification, reading comprehension, sequence tagging, etc.).  This should make
it easy to see if something you are trying to do is already implemented in DeepQA or not.

