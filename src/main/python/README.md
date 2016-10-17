# Structure of the python code

The python code is structured such that the `deep_qa` module contains all of the library code, with
no entry points (`main() methods`) within that module.  Instead, all scripts that use this library
code live here, in this directory, outside of the `deep_qa` module.  This is [standard
practice](http://stackoverflow.com/questions/14132789/relative-imports-for-the-billionth-time/14132912#14132912)
for python code.  (Note that there are a couple of auxiliary modules that are dealing with data
processing that don't yet follow this guideline: the `sentence_corruption` and `span_prediction`
modules.  When we get the bandwidth to fix those, they will be updated to work the same way.)

There's also a `proto` module in here.  You almost certainly don't need to worry about it.  In
order to integrate this code with the Aristo system, we need to run a python server that a scala
client can talk to, and that communication is done using gRPC and protobuf messages.  The `proto`
module is automatically generated from the protobuf files in `src/main/protobuf/`, but we check it
in here so that our continuous integration server doesn't have to know about commands to build
protobuf libraries (we don't expect the protobuf files to change much, anyway).
