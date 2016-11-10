This package contains code that actually runs experiments, typically by defining a bunch of
parameters, then calling runPipeline() on some `Step` object defined in a package under `pipeline/`.

For running a question answering experiment with a deep neural network, there are two possible
entry points to this code.  The first is
[`NeuralNetworkTrainerStep`](../pipeline/NeuralNetworkTrainerStep.scala).  This `Step` takes a set
of parameters for a model and a dataset, trains a model, and reports validation accuracy (more
detail on that below).  If you call `runPipeline()` on this `Step` directly, it will _always_ train
the model, no matter if it has been done before and we already have saved results for this model.

The second, and better, entry point to this code is to call an `Evaluator` `Step`.  An `Evaluator`
takes a collection of model specifications that would have been inputs to a
`NeuralNetworkTrainerStep`, runs all of the models, and outputs a table of results.  Calling
`runPipeline()` on an `Evaluator` will _not_ train a model if it finds a saved result that matches
the parameters of the experiment you're trying to run - it'll just use the saved result, saving you
a bunch of time, and avoiding duplicating work between experimenters.  The code assumes that you
are running from an EC2 instance, with our (AI2) shared EFS volume mounted, so that you have access
to `/efs/data/dlfa/`.  Much of the data used in our experiments is already stored there, and models
and results are saved there after running experiments.  If you need to run this from another
machine, or you're not at AI2, you'll currently have to change a bunch of paths in the code.
Sorry.

What follows is a short introduction to how to construct parameter objects for use with these two
`Steps`.

### Parameters to `NeuralNetworkTrainerStep`

There are a few ways to figure out how to construct a parameter object to pass to a
`NeuralNetworkTrainerStep`.  One of them is to just look at the experiments that use the `Step`,
like [`MemoryNetworkExperiments`](MemoryNetworkExperiments.scala), or
[`BabiExperiments`](BabiExperiments.scala).  You could also trace through the code in
`NeuralNetworkTrainerStep`, looking at where the parameters are read, and following through to
dependent classes, and so on.  This is currently the only way to get a really comprehensive look
at the available options.

Since `NeuralNetworkTrainerStep` is the most likely use case, though, I'll give a third option here
and briefly outline all of the parameters that go into this `Step`, and what some options are.
Note, though, that this documentation might get out of date, and the links might break.  If you
notice something is out of date, open an issue or submit a pull request.

`NeuralNetworkTrainerStep` has four [valid parameters](../pipeline/NeuralNetworkTrainerStep.scala#L31-L35):

- `model params`: This defines the model architecture, training, pre-training, and everything else
  that will go into the python neural network code, except for the data to use.  These parameters
get passed directly to the python code, so if you want to know what options are available here, go
look at the python code, or look at the examples in the [`example_experiments`
directory](../../../../../../../example_experiments)

- `name`: This parameter is optional, and is only used when showing results.  If you want to give
  a name to an experiment, e.g., to distinguish between model variants in a particular experiment,
you can do that here, and it will show up in the tables that we produce.

- `dataset`: This parameter defines the training data used in the experiment.  These parameters
  will get passed to a `DatasetStep`, which will create whatever files are necessary for the
dataset.  For example, if you're running on the bAbI dataset, this `Step` will convert the original
bAbI files into the format the DeepQA code can read.  If you're running on science questions,
using a corpus in an elastic search index to retrieve background information, this `Step` will
convert the questions into the right format and perform searches over the index to construct a
file with the background information.  Below we'll look at these parameters in a little more
detail.

- `validation dataset`: This parameter is optional.  If it's left blank, Keras will split the
  training dataset into train/dev, and that's all (this is configurable with `model params`; see
the python code for details).  If this is given, we'll use all of the training dataset for
training, and use this dataset for validation.

### Parameters to `DatasetStep`

`DatasetStep` has just one [valid parameter](../pipeline/DatasetStep.scala#L21):

- `data files`: This is a list of `SentenceProducer` parameters.  A `Dataset` consists of a
  set of sentence files, and each entry in this `data files` list specifies one of these files.
For example, one file might be a set of science questions that have been processed in some
particular way, and another file might have background knowledge associated with each of the
processed questions.


This brings us to:

### Parameters to `SentenceProducer` `Steps`

There are a lot of different `SentenceProducers`, and they all take different parameters.  All of
them, however, have a few common parameters, which you can see in the [`baseParams`
field](../pipeline/SentenceProducer.scala#L33) of the `SentenceProducer` trait:

- `sentence producer type`: This specifies the class that will handle producing these sentences.
  A few options are `background searcher`, which takes as input a set of sentences (which in turn
have been produced by some `SentenceProducer`) and a corpus, and finds relevant background from
the corpus for each fo the questions; and `question interpreter`, which takes a multiple choice
question file and produces a list of sentences, in a few different possible formats.  For a
complete list of options, see
[`SentenceProducer.create()`](../pipeline/SentenceProducer.scala#L59).

- `create sentence indices`: This specifies whether the produced sentences have associated
  indices.  If you are producing multiple files that are correlated in some way, you probably need
to set this to `true` (the `background searchers` require that this is `true` for
`SentenceProducers` that they depend on, for instance, so that they can link the background they
retrieve to the correct sentence).

- `max sentences`: Instructs the `SentenceProducer` to truncate its results to some max number of
  sentences.

### Pre-specified datasets

For some popular datasets, we have some parameter specifications already built, to make experiments
with these datasets easier.  You can see these in the [`datasets`](datasets) subdirectory here.
For example, we have several datasets from Facebook AI Research specified in
[`datasets/Facebook.scala`](datasets/Facebook.scala), and you can access them in experiments by
using the member variables of the `BabiDatasets` and `ChildrensBookDatasets` objects.  We
similarly have the SNLI corpus available, and several different ways of constructing science
question answering datasets, with various parameter settings.  The datasets available here will
grow over time.  Note that if you're not in an environment where you have access to AI2's EFS
volume (the path `/efs/data/dlfa/`), you'll have to change path names in the code to reflect where
you put the downloaded data for these datasets.

### Pre-specified models

We've also tried to make specifying models in experiments easier, with some pre-specified
parameter objects in [`Models.scala`](Models.scala).  Here you can find parameters for various
encoders, debug settings, training times, pre-training, and basic model structures.  You can pick
pieces that you want and put them together into a final experiment.  An example of how to do this
is in [`MemoryNetworkExperiments`](MemoryNetworkExperiments.scala#L15-L22).


### A specific `SentenceProducer` example

This is an old example, but it might help you get more intuition for how the pipeline code works.
Above, we recommend calling `runPipeline()` on an `Evaluator` `Step`, but you could run it on any
`Step` that you want to, if you just want to run all of the steps up to some part of a pipeline.
This example shows how to just create some dataset files.

You can see how data files are prepared by looking in (an old version of)
[MemoryNetworkExperiments.scala](https://github.com/allenai/deep_qa/blob/089954f5713b4b91d5a0b73c375d5d2983383772/src/main/scala/org/allenai/deep_qa/experiments/MemoryNetworkExperiments.scala).
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
