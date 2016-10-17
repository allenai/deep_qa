[![Build Status](https://semaphoreci.com/api/v1/projects/b3480192-615d-4981-ba34-62afeb9d9ae6/953929/shields_badge.svg)](https://semaphoreci.com/allenai/deep_qa)

# Deep QA

This repository contains code for training deep learning systems to do question answering tasks.
Our primary focus is on Aristo's science questions, though we can run various models on several
popular datasets.

This code is a mix of scala (for data processing / pipeline management) and python (for actually
training and executing deep models with Keras / Theano / TensorFlow).

# Implemented models

This repository implements several variants of memory networks, including the models found in these papers:

- The original MemNN, from [Memory Networks](https://arxiv.org/abs/1410.3916), by Weston, Chopra and Bordes
- [End-to-end memory networks](https://www.semanticscholar.org/paper/End-To-End-Memory-Networks-Sukhbaatar-Szlam/10ebd5c40277ecba4ed45d3dc12f9f1226720523), by Sukhbaatar and others (close, but still in progress)
- [Dynamic memory networks](https://www.semanticscholar.org/paper/Ask-Me-Anything-Dynamic-Memory-Networks-for-Kumar-Irsoy/04ee77ef1143af8b19f71c63b8c5b077c5387855), by Kumar and others (close, but still in progress)
- DMN+, from [Dynamic Memory Networks for Visual and Textual Question Answering](https://www.semanticscholar.org/paper/Dynamic-Memory-Networks-for-Visual-and-Textual-Xiong-Merity/b2624c3cb508bf053e620a090332abce904099a1), by Xiong, Merity and Socher (close, but still in progress)
- The attentive reader, from [Teaching Machines to Read and Comprehend](https://www.semanticscholar.org/paper/Teaching-Machines-to-Read-and-Comprehend-Hermann-Kocisk%C3%BD/2cb8497f9214735ffd1bd57db645794459b8ff41), by Hermann and others
- Windowed-memory MemNNs, from [The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations](https://www.semanticscholar.org/paper/The-Goldilocks-Principle-Reading-Children-s-Books-Hill-Bordes/1ee46c3b71ebe336d0b278de9093cfca7af7390b) (in progress)

As well as some of our own, as-yet-unpublished variants.  There is a lot of similarity between the models in these papers, and our code is structured in a way to allow for easily switching between these models.

# Datasets

This code allows for easy experimentation with the following datasets:

- [AI2 Elementary school science questions (no diagrams)](http://allenai.org/data.html)
- [The Facebook Children's book test dataset](https://research.facebook.com/research/babi/#cbt)
- [The Facebook bAbI dataset](https://research.facebook.com/research/babi/)

And more to come...  In the near future, we hope to also include easy experimentation with
[SQuAD](https://rajpurkar.github.io/SQuAD-explorer/), [CNN/Daily
Mail](http://cs.nyu.edu/~kcho/DMQA/), and
[SimpleQuestions](https://research.facebook.com/research/babi/).

# Usage Guide

This code is a mix of scala and python.  The intent is that the data processing and experiment
pipeline code is in scala, and the deep learning code is in python.  Eventually (hopefully soon),
there will be an easy way to specify a set of experiments entirely in scala and run them from sbt
with a single command.  That's not quite ready yet, though, so you currently have to set up the
data in scala, set up the deep learning model with a json configuration file, then use sbt to
create the data files and python to do actual training.

## Running experiments with scala

This code is still a work in progress, but you can see how data files are prepared by looking in
[MemoryNetworkExperiments.scala](https://github.com/allenai/deep_qa/blob/master/src/main/scala/org/allenai/deep_qa/experiments/MemoryNetworkExperiments.scala).
That class sets up a JSON object that gets passed to a pipeline Step, which will run the pipeline,
creating whatever intermediate files it needs.

For example, if you wanted to create the necessary input files to run an experiment with Aristo's
science questions, starting from a raw question file from [AI2's
webpage](http://allenai.org/data.html) and a pointer to an Elastic Search index containing a
science corpus, you would set up the parameters required for the last step in that pipeline (doing
a search over the corpus with all of the processed questions), and run it using a command like
[this](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/experiments/MemoryNetworkExperiments.scala#L188).
That code creates a `BackgroundCorpusSearcherStep`, taking
[parameters](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/experiments/MemoryNetworkExperiments.scala#L123)
that identify the corpus to be searched and the sentences that should be used for the search.  The
[sentence
parameters](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/experiments/MemoryNetworkExperiments.scala#L108),
in turn, specify how those sentences were produced (in this case, they were obtained from a raw
Aristo question file, by appending each answer option to the text of the original question to
obtain a list of sentences).  The pipeline code takes care of running all intermediate `Steps` to
get from the original input file to the final output.  These `Steps` take their inputs and outputs
from the file system, so intermediate results are saved across runs and are only ever constructed
once for each unique combination of parameters.  Each `Step` specifies the files it needs as inputs
and the files it produces as outputs; if necessary input files are not present, and you've
specified how to create them using another `Step`, the pipeline code will run those prior `Steps`
before running the current one.  You can see how this works by looking at the
[code](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/pipeline/BackgroundCorpusSearcherStep.scala#L94)
for the
[`BackgroundCorpusSearcherStep`](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/pipeline/BackgroundCorpusSearcherStep.scala#L51-L60).

You might notice that there's a
[`NeuralNetworkTrainer`](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/pipeline/NeuralNetworkTrainer.scala)
step in the code, also.  This `Step` runs the python code, though it's currently in a broken
state.  Eventually, you will just specify the model parameters you want, similar to how you do with
the data processing, then run this `Step`, and it will create all of the necessary input files for
the model you want to train, and train the model for you, with a single command.

## Running experiments with python

Because the scala code isn't ready to run end-to-end yet, you currently have to run the python code
manually.  To do this, from the base directory, you run the command `python
src/main/python/run_solver.py [model_config]`.  You must use python >= 3.5, as we make heavy use of
the type annotations introduced in python 3.5 to aid in code readability (I recommend using
[anaconda](https://www.continuum.io/downloads) to set up python 3, if you don't have it set up
already).

You can see some examples of what model configuration files look like in the [example
experiments directory](https://github.com/allenai/deep_qa/tree/master/example_experiments).  We
try to keep these up to date, but the way parameters are specified is still sometimes in a state
of flux, so we make no promises that these are actually usable with the current master (and you'll
have to provide your own input files to use them, in any event).  Looking at the most recently
added or changed example experiment should be your best bet to get an accurate format.  And if you
find one that's out of date, submitting a pull request to fix it would be really nice!

The best way currently to get an idea for what options are available in this configuration file,
and what those options mean, is to look at the class mentioned in the `solver_class` field.
Looking at the
[`dynamic_memory_network.json`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/example_experiments/dynamic_memory_network.json)
example, we can see that it's using a `MemoryNetworkSolver` as it's
[`solver_class`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/example_experiments/dynamic_memory_network.json#L2).
If we go to that class's [`__init__`
method](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/with_memory/memory_network.py#L69)
in the code, we can see the parameters that it takes: things like `num_memory_layers`,
`knowledge_encoder`, `entailment_model`, and so on.  If you contine on to its super class,
[`TextTrainer`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/training/text_trainer.py#L32),
you'll find more parameters, this time for things that deal with word embeddings and sentence
encoders.  Finally, you can continue to the base class,
[`Trainer`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/training/text_trainer.py#L32),
to see parameters for things like whether and where models should be saved, how to run training,
specifying debug output, running pre-training, and other things.  It would be nice to automatically
generate some website to document all of these parameters, but I don't know how to do that and
don't have the time to dedicate to making it happen.  So for now, just read the comments that are
in the code.

There are several places where we give lists of available choices for particular options.  For
example, there is a [list of concrete
solver classes](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/__init__.py#L15-L24)
that are valid options for the `solver_class` parameter in a model config file.  One way to find
lists of available options for these parameters (other than just by tracing the handling of
parameters in the code) is by searching github for
[`get_choice`](https://github.com/allenai/deep_qa/search?utf8=%E2%9C%93&q=get_choice) or
[`get_choice_with_default`](https://github.com/allenai/deep_qa/search?utf8=%E2%9C%93&q=get_choice_with_default).
This might point you, for instance, to the
[`knowledge_encoders`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/with_memory/memory_network.py#L217)
field in `memory_network.py`, which is
[imported](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/with_memory/memory_network.py#L17)
from `layers/knowledge_encoders.py`, where it is defined at the [bottom of the
file](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/layers/knowledge_encoders.py#L75-L77).
In general, the places where there are these kinds of options are in the solver class (already
mentioned), and the various layers we have implemented - each kind of `Layer` will typically
specify a list of options either at the bottom of the corresponding file, or in an associated
`__init__.py` file (as is done with the [sentence
encoders](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/layers/encoders/__init__.py)).

We've tried to also give reasonable documentation throughout the code, both in docstring comments
and in READMEs distributed throughout the code packages, so browsing github should be pretty
informative if you're confused about something.  If you're still confused about how something
works, open an issue asking to improve documentation of a particular piece of the code (or, if
you've figured it out after searching a bit, submit a pull request containing documentation
improvements that would have helped you).

# Contributing

If you use this code and think something could be improved, pull requests are very welcome.
Opening an issue is ok, too, but we're a lot more likely to respond to a PR.  The primary
maintainer of this code is [Matt Gardner](https://matt-gardner.github.io/), with a lot of help
from [Pradeep Dasigi](http://www.cs.cmu.edu/~pdasigi/) (who was the initial author of this
codebase) and [Mark Neumann](http://markneumann.xyz/).

# License

This code is released under the terms of the [Apache 2 license](https://www.apache.org/licenses/LICENSE-2.0).
