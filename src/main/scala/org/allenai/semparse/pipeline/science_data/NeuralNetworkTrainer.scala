package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

/**
 * This Step executes code in a subprocess to train a neural network using some deep learning
 * library.
 *
 * There are lots of different neural network architectures we could try, some of which take very
 * different kinds of input.  So, this is an abstract class, so that subclasses can define their
 * own input requirements.  Then we take those inputs and pass them, along with any neural network
 * parameters, to the neural network code by opening a subprocess.  In general, you should just
 * define one subclass here for each type of required input (e.g., just positive and negative
 * sentences; positive and negative sentences plus background sentences; positive and negative
 * sentences plus KB features; ...), then use the parameters of that subclass to change the neural
 * network as necessary.
 */
abstract class NeuralNetworkTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  override val binary = "python"
}

object NeuralNetworkTrainer {
  def create(params: JValue, fileUtil: FileUtil): NeuralNetworkTrainer = {
    // TODO(matt): actually have a type parameter, and use it here.
    new NoBackgroundKnowledgeNNTrainer(params, fileUtil)
  }
}

class NoBackgroundKnowledgeNNTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  override val name = "Neural Network Trainer, no background knowledge input"

  val validParams = Seq("positive data", "negative data", "validataion questions",
    "number of epochs")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val numEpochs = JsonHelper.extractWithDefault(params, "number of epochs", 20)
  val trainingFile = ""  // TODO(matt)
  val validationFile = ""  // TODO(matt)

  override val scriptFile = "src/main/python/prop_scorer/prop_scorer.py"
  override val arguments = Seq[String](
    "--train_file", trainingFile,  // TODO(matt): change prop_scorer.py to take two separate input files
    "--validation_file", validationFile,
    "--use_tree_lstm", false.toString,  // TODO(matt)
    "--length-upper-limit", 100.toString,  // TODO(matt)
    "--max_train_size", 1000000.toString,  // TODO(matt)
    "--num_epochs", numEpochs.toString
  )

  // TODO(matt): positive training data, negative training data, validation questions
  override val inputs: Set[(String, Option[Step])] = Set()

  // TODO(matt): model file, other things that get saved
  override val outputs: Set[String] = Set()

  // TODO(matt): base these off of the model file
  override val inProgressFile = ""  // TODO(matt)
  override val paramFile = ""  // TODO(matt)
}
