# Models

In this module we define a number of concrete models.  The models are grouped by task, where each
task has a roughly coherent input/output specification.  See the README in each submodule for a
description of the task models in that submodule are designed to solve.

You should think of these models as more of "model families" than actual models, though, as there
are typically options left unspecified in the models themselves.  For example, models in this
module might have a layer that encodes word sequences into vectors; they just call a method on
`TextTrainer` to get an encoder, and the decision for which actual encoder is used (an LSTM, a
CNN, or something else) happens in the parameters passed to `TextTrainer`.  If you really want to,
you can hard-code specific decisions for these things, but most models we have here use the
`TextTrainer` API to abstract away these decisions, giving implementations of a class of similar
models, instead of a single model.

We also define a few general `Pretrainers` in a submodule here.  The `Pretrainers` in this
top-level submodule are suitable to pre-train a large class of models (e.g., any model that
encodes sentences), while more task-specific `Pretrainers` are found in that task's submodule.
