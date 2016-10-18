# Top Level Design

The below diagram gives a top level abstraction of what comprises a memory network. It follows several basic steps:
* Encode the Question and Background Knowledge using a sentence encoder. This might be an LSTM, BOW, or something more complex, such as a TreeLSTM.
<img align="right" height="400" width="440" src="../../../../../../doc/module_breakdown.png">

* Given the encoded Question, Background Knowledge and an initialised Memory, proceed to do several "Memory hops". This process includes:
  * Attend over the Background Knowledge. This could be a simple dot product similarity with the Memory vector, or a more complicated parameterized attention function.
  * Combine the generated attention with the background knowledge. As standard, this would be a weighted average, but other, more complex ways of combining these are possible.
  * Using the attended knowledge, question and memory, update the memory vector for input to the next memory network hop.
* After X number of memory network hops, create a vector for classification/entailment. This could be just the final memory step, or something more, such as the original question.
* Now, depending on the type of Solver, we use different types of final layer to get the required output. For instance, a MultipleChoiceMemoryNetworkSolver will use a softmax to generate a distribution over the possible answer candidates.


Hopefully the above gives a clear way to approach this library.
