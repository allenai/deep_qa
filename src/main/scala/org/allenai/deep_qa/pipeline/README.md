Pipeline code defines Steps using my [pipeline library](https://github.com/matt-gardner/util) that
are put together to run experiments.  Each Step takes some inputs and produces some outputs that
live on the file system.  The pipeline library lets you chain together a bunch of steps and ask for
the output of the last step that you care about.  The library will then run whatever pre-requisite
steps haven't already been run, logging whatever parameters are used to produce all of the output
files for each step.

In general, Steps should be small objects, mostly calling out to classes that live elsewhere that
implement the actual logic of what's going on in the Step.  The code in here wasn't originally
written that way, and still needs to be migrated to match this preferred design.
