package org.allenai.semparse.pipeline.science_data

import org.json4s._

import com.mattg.pipeline.Step
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
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  val baseParams = Seq("command to execute")

  val commandToExecute = (params \ "command to execute").extract[String]

  override def _runStep() {
    // TODO(matt): execute the command here
  }
}


class NoBackgroundKnowledgeNNTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  override val name = "Neural Network Trainer, no background knowledge input"

  override val inputs: Set[(String, Option[Step])] = Set()  // TODO(matt)
  override val outputs: Set[String] = Set()  // TODO(matt)
  override val inProgressFile = ""  // TODO(matt)
  override val paramFile = ""  // TODO(matt)
}


object NeuralNetworkTrainer {
  def create(params: JValue, fileUtil: FileUtil): NeuralNetworkTrainer = {
    // TODO(matt)
    new NoBackgroundKnowledgeNNTrainer(params, fileUtil)
  }
}
