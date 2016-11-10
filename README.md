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
For a description of how we've built an extensible memory network architecture in this library, see [this readme.](./src/main/python/deep_qa/solvers/with_memory/README.md)
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
pipeline code is in scala, and the deep learning code is in python.  The recommended approach is to
set up your experiments in scala code, then run them through `sbt`.  Some documentation on how to
do this is found in the [README for the `org.allenai.deep_qa.experiments`
package](src/main/scala/org/allenai/deep_qa/experiments/).

## Running experiments with python

If for whatever reason you don't want to gain the benefits of the scala pipeline when running
experiments, you can run the python code manually.  To do this, from the base directory, you run
the command `python src/main/python/run_solver.py [model_config]`.  You must use python >= 3.5, as
we make heavy use of the type annotations introduced in python 3.5 to aid in code readability (I
recommend using [anaconda](https://www.continuum.io/downloads) to set up python 3, if you don't
have it set up already).

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
example, we can see that it's using a `MultipleTrueFalseMemoryNetworkSolver` as it's
[`solver_class`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/example_experiments/dynamic_memory_network.json#L2).
If we go to that class's [`__init__`
method](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/with_memory/multiple_true_false_memory_network.py#L31),
in the code, we don't see any parameters, because `MultipleTrueFalseMemoryNetworkSolver` has no
unique parameters of its own.  So, we continue up the class hierarchy to
[`MemoryNetworkSolver`](https://github.com/allenai/deep_qa/blob/932849e8b3ebec6882680231924248669cc19758/src/main/python/deep_qa/solvers/with_memory/memory_network.py#L69),
and we can see the parameters that it takes: things like `num_memory_layers`, `knowledge_encoder`,
`entailment_model`, and so on.  If you continue on to its super class,
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
