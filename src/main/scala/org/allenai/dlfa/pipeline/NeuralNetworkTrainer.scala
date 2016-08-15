package org.allenai.dlfa.pipeline

import org.json4s._

import com.mattg.pipeline.Step
import com.mattg.pipeline.SubprocessStep
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

object NeuralNetworkTrainer {
  def create(params: JValue, fileUtil: FileUtil): NeuralNetworkTrainer = {
    implicit val formats = DefaultFormats
    (params \ "model type").extract[String] match {
      case "simple lstm" => new SimpleLstmTrainer(params, fileUtil)
      case "memory network" => new MemoryNetworkTrainer(params, fileUtil)
      case t => throw new IllegalStateException(s"Unrecognized neural network model type: $t")
    }
  }
}

/**
 * This Step executes code in a subprocess to train a neural network using some deep learning
 * library.
 *
 * There are lots of different neural network architectures we could try, some of which take very
 * different kinds of input.  So, this is an abstract class, so that subclasses can define their
 * own input requirements.  Then we take those inputs and pass them, along with any neural network
 * parameters, to the neural network code by opening a subprocess.
 *
 * The current design is to have one class here for each class under solvers/ in the python code.
 * This base class corresponds to the NNSolver base class, and handles parameters for that base
 * class.
 */
abstract class NeuralNetworkTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends SubprocessStep(Some(params), fileUtil) {
  implicit val formats = DefaultFormats

  // Anything that is common to training _all_ neural network models should go here.  Data-specific
  // parameters go in subclasses.
  val baseParams = Seq(
    "model type",
    "model name",
    "positive data",
    "negative data",
    "validation questions",
    "number of epochs",
    "max training instances",
    "max sentence length",
    "embedding size",
    "embedding dropout",
    "pretrained embeddings"
  )

  val numEpochs = JsonHelper.extractWithDefault(params, "number of epochs", 20)

  val embeddingSize = JsonHelper.extractWithDefault(params, "embedding size", 50)
  val embeddingDropout = JsonHelper.extractWithDefault(params, "embedding dropout", 0.5)
  val pretrainedEmbeddingArgs = (params \ "pretrained embeddings") match {
    case JNothing => Seq()
    case jval => {
      val allowedArgs = Seq("file", "fine tune", "add projection")
      JsonHelper.ensureNoExtras(jval, "pretrained embeddings", allowedArgs)
      val embeddingsFile = (jval \ "file").extract[String]
      val fineTune = JsonHelper.extractWithDefault(jval, "fine tune", false)
      val addProjection = JsonHelper.extractWithDefault(jval, "add projection", true)
      val fileArgs = Seq("--pretrained_embeddings_file", embeddingsFile)
      val fineTuneArg = if (fineTune) Seq("--fine_tune_embeddings") else Seq()
      val addProjectionArg = if (addProjection) Seq("--project_embeddings") else Seq()
      fileArgs ++ fineTuneArg ++ addProjectionArg
    }
  }

  val maxTrainingInstances = JsonHelper.extractAsOption[Int](params, "max training instances")
  val maxTrainingInstancesArgs =
    maxTrainingInstances.map(max => Seq("--max_training_instances", max.toString)).toSeq.flatten

  val modelName = (params \ "model name").extract[String]
  val modelPrefix = s"models/$modelName"  // TODO(matt): make this a function of the arguments?

  val maxSentenceLength = JsonHelper.extractAsOption[Int](params, "max sentence length")
  val maxSentenceLengthArgs =
    maxSentenceLength.map(max => Seq("--length_upper_limit", max.toString)).toSeq.flatten

  val positiveDataProducer = SentenceProducer.create(params \ "positive data", fileUtil)
  val positiveTrainingFile = positiveDataProducer.outputFile
  val positiveDataInput = (positiveTrainingFile, Some(positiveDataProducer))

  val negativeDataProducer = SentenceProducer.create(params \ "negative data", fileUtil)
  val negativeTrainingFile = negativeDataProducer.outputFile
  val negativeDataInput = (negativeTrainingFile, Some(negativeDataProducer))

  val questionInterpreter = new QuestionInterpreter(params \ "validation questions", fileUtil)
  val validationQuestionsFile = questionInterpreter.outputFile
  val validationInput = (validationQuestionsFile, Some(questionInterpreter))

  override val binary = "python"
  override val scriptFile = Some("src/main/python/run_solver.py")

  val baseInputs = Set[(String, Option[Step])](positiveDataInput, negativeDataInput, validationInput)

  val baseArguments = Seq[String](
    "--positive_train_file", positiveTrainingFile,
    "--negative_train_file", negativeTrainingFile,
    "--validation_file", validationQuestionsFile,
    "--num_epochs", numEpochs.toString,
    "--model_serialization_prefix", modelPrefix
  ) ++ maxTrainingInstancesArgs ++ maxSentenceLengthArgs ++ pretrainedEmbeddingArgs

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

/**
 * This NeuralNetworkTrainer is a simple LSTM, corresponding to lstm_solver.py.  We take good and
 * bad sentences as input, and nothing else.  The LSTM is trained to map good sentences to "true"
 * and bad sentences to "false".  This is mostly here as a baseline, as we expect to need
 * additional input to actual do well at answering science questions.  At best, this kind of a
 * model will look for word correlations to decide whether a sentence is true or false, similar to
 * what the Salience solver does.
 */
class SimpleLstmTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  override val name = "Neural Network Trainer, on sentences, no background knowledge input"

  val validParams = baseParams
  JsonHelper.ensureNoExtras(params, name, validParams)

  override val arguments = Seq[String](
    "LSTMSolver",
    "--validation_file", validationQuestionsFile,
    "--num_epochs", numEpochs.toString
  ) ++ baseArguments

  override val inputs: Set[(String, Option[Step])] = baseInputs
  override val outputs = modelOutputs.toSet

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_params.json"
}

/**
 * This NeuralNetworkTrainer is a memory network, corresponding to memory_network.py.  We take good
 * and bad sentences as input, as well as a collection of background sentences for each input.
 */
class MemoryNetworkTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  override val name = "Neural Network Trainer, on sentences, no background knowledge input"

  val validParams = baseParams ++ Seq(
    "positive data",
    "positive background",
    "negative data",
    "negative background",
    "validation background",
    "memory layer type",
    "num memory layers",
    "max sentence length"
  )
  JsonHelper.ensureNoExtras(params, name, validParams)

  val positiveBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "positive background", fileUtil)
  val positiveBackgroundFile = positiveBackgroundSearcher.outputFile

  val negativeBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "negative background", fileUtil)
  val negativeBackgroundFile = negativeBackgroundSearcher.outputFile

  val validationBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "validation background", fileUtil)
  val validationBackgroundFile = validationBackgroundSearcher.outputFile

  val numMemoryLayers = JsonHelper.extractWithDefault(params, "num memory layers", 1)

  val choices = Seq("attentive", "memory")
  val defaultChoice = choices(0)
  val memoryLayerType = JsonHelper.extractChoiceWithDefault(params, "memory layer type", choices, defaultChoice)

  override val arguments = Seq[String](
    "MemoryNetworkSolver",
    "--positive_train_background", positiveBackgroundFile,
    "--negative_train_background", negativeBackgroundFile,
    "--validation_background", validationBackgroundFile,
    "--memory_layer", memoryLayerType,
    "--num_memory_layers", numMemoryLayers.toString
  ) ++ baseArguments

  override val inputs: Set[(String, Option[Step])] = baseInputs ++ Set(
    (positiveBackgroundFile, Some(positiveBackgroundSearcher)),
    (negativeBackgroundFile, Some(negativeBackgroundSearcher)),
    (validationBackgroundFile, Some(validationBackgroundSearcher))
  )
  override val outputs = modelOutputs.toSet

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_params.json"
}
