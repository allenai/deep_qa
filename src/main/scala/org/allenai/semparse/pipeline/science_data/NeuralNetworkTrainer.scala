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
  implicit val formats = DefaultFormats

  // Anything that is common to training _all_ neural network models should go here.  Data-specific
  // parameters go in subclasses.
  val baseParams = Seq("model type", "model name", "validation questions", "number of epochs",
    "max training instances")

  // The way these parameters work assumes some commonality between the underlying python scripts
  // that train models.  Currently there's only one script, so that's ok...
  val numEpochs = JsonHelper.extractWithDefault(params, "number of epochs", 20)
  val maxTrainingInstances = JsonHelper.extractAsOption[Int](params, "max training instances")
  val maxTrainingInstancesArgs =
    maxTrainingInstances.map(max => Seq("--max_train_size", max.toString)).toSeq.flatten

  val modelName = (params \ "model name").extract[String]
  val modelPrefix = s"models/$modelName"

  val questionInterpreter = new QuestionInterpreter(params \ "validation questions", fileUtil)
  val validationQuestionsFile = questionInterpreter.outputFile
  val validationInput = (validationQuestionsFile, Some(questionInterpreter))

  override val binary = "python"

  // These three outputs are written by the python code.  The config.json file is a specification
  // of the model layers, so that keras can reconstruct the saved model object / computational
  // graph.  The weights.h5 file is a specification of all of the neural network weights for the
  // model specified by the config.json file.  The data_indexer.pkl file is a pickled DataIndexer,
  // which maps words to indices.
  val modelOutputs = Seq(
    modelPrefix + "_weights.h5",
    modelPrefix + "_data_indexer.pkl",
    modelPrefix + "_config.json"
  )

}

object NeuralNetworkTrainer {
  def create(params: JValue, fileUtil: FileUtil): NeuralNetworkTrainer = {
    implicit val formats = DefaultFormats
    (params \ "model type").extract[String] match {
      case "simple lstm" => new NoBackgroundKnowledgeNNSentenceTrainer(params, fileUtil)
      case t => throw new IllegalStateException(s"Unrecognized neural network model type: $t")
    }
  }
}

/**
 * This NeuralNetworkTrainer is a simple LSTM.  We take good and bad sentences as input, and
 * nothing else.  The LSTM is trained to map good sentences to "true" and bad sentences to
 * "false".  This is mostly here as a baseline, as we expect to need additional input to actual do
 * well at answering science questions.  At best, this kind of a model will look for word
 * correlations to decide whether a sentence is true or false, similar to what the Salience solver
 * does.
 */
class NoBackgroundKnowledgeNNSentenceTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  override val name = "Neural Network Trainer, on sentences, no background knowledge input"

  val validParams = baseParams ++ Seq("positive data", "negative data", "max sentence length")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val maxSentenceLength = JsonHelper.extractAsOption[Int](params, "max sentence length")
  val maxSentenceLengthArgs =
    maxSentenceLength.map(max => Seq("--length_upper_limit", max.toString)).toSeq.flatten

  val positiveDataProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveTrainingFile = positiveDataProducer.outputFile
  val positiveDataInput = (positiveTrainingFile, Some(positiveDataProducer))

  val negativeDataProducer = (params \ "negative data") match {
    case JNothing => None
    case jval => Some(SentenceProducer.create(jval, fileUtil))
  }
  val negativeDataArgs =
    negativeDataProducer.map(p => Seq("--negative_train_file", p.outputFile)).toSeq.flatten
  val negativeDataInput = negativeDataProducer.map(p => (p.outputFile, Some(p))).toSet

  override val scriptFile = Some("src/main/python/prop_scorer/nn_solver.py")
  override val arguments = Seq[String](
    "--positive_train_file", positiveTrainingFile,
    "--validation_file", validationQuestionsFile,
    "--num_epochs", numEpochs.toString,
    "--model_serialization_prefix", modelPrefix
  ) ++ negativeDataArgs ++ maxTrainingInstancesArgs ++ maxSentenceLengthArgs

  override val inputs: Set[(String, Option[Step])] = Set(
    positiveDataInput,
    validationInput
  ) ++ negativeDataInput
  override val outputs = modelOutputs.toSet

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_params.json"
}
