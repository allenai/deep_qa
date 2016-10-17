# The DeepQA library

I think there are two main contributions that this library makes:

1. We provide a nice interface to training, validating, and debugging Keras models.  Instead of
   writing code to run an experiment, you just specify a JSON file that describes your experiment,
and our library will run it for you (assuming you only want to use the components that we've
implemented).
2. We've implemented several variants of memory networks, neural network architectures that
   attempt a kind of reasoning over background knowledge, that are not trivial to implement.
We've done this in a way that is configurable and extendable, such that you can easily run
experiments with several different memory network variants just by changing some parameters in a
configuration file, or implement a new idea just by writing a new component.

This library has several main components:

- A `training` module, which has a bunch of helper code for training Keras models of various kinds
- A `solvers` module, specifying particular Keras models for question answering (in Aristo, we
  call a question answering system a "solver", which is where the name comes from).
- A `layers` module, which contains code for custom Keras `Layers` that we have written.
- A `data` module, containing code for reading in data from files and converting it into numpy
  arrays suitable for use with Keras.
- A `common` module, which has a few random things dealing with reading parameters and a few other
  things.

There are also a couple of components that are not yet integrated into the DeepQA library, but
exist here in an initial state, waiting to be cleaned up and fully integrated.  These are the
`sentence_corruption` module and the `span_prediction` module.  At this point, these modules are
totally independent from the rest of the library.  (Actually, it would probably be a good idea to
move them outside of `deep_qa`, to make this more clear... TODO(matt).)
