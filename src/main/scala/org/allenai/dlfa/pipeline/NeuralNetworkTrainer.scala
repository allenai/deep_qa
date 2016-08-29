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
      case "memory network with differentiable search" => new DifferentiableSearchTrainer(params, fileUtil)
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
    "patience",
    "number of epochs",
    "max training instances",
    "max sentence length",
    "embedding size",
    "embedding dropout",
    "pretrained embeddings"
  )

  val numEpochs = JsonHelper.extractWithDefault(params, "number of epochs", 20)
  val patience = JsonHelper.extractWithDefault(params, "patience", 1)

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
    "--patience", patience.toString,
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
 * Because I had a hard time getting parameter validation to work right when I made
 * DifferentiableSearchSolver inherit directly from MemoryNetworkTrainer, I refactored the common
 * stuff out into this base class, which both of those two now inherit from.
 */
abstract class MemoryNetworkBase(
  params: JValue,
  fileUtil: FileUtil
) extends NeuralNetworkTrainer(params, fileUtil) {
  val memoryNetworkParams = baseParams ++ Seq(
    "positive background",
    "negative background",
    "validation background",
    "knowledge selector",
    "memory updater",
    "entailment model",
    "num memory layers"
  )

  val positiveBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "positive background", fileUtil)
  val positiveBackgroundFile = positiveBackgroundSearcher.outputFile

  val negativeBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "negative background", fileUtil)
  val negativeBackgroundFile = negativeBackgroundSearcher.outputFile

  val validationBackgroundSearcher = BackgroundCorpusSearcher.create(params \ "validation background", fileUtil)
  val validationBackgroundFile = validationBackgroundSearcher.outputFile

  val numMemoryLayers = JsonHelper.extractWithDefault(params, "num memory layers", 1)

  val knowledgeSelector = (params \ "knowledge selector").extract[String]
  val memoryUpdater = (params \ "memory updater").extract[String]
  val entailmentModel = (params \ "entailment model").extract[String]

  val memoryNetworkArguments = baseArguments ++ Seq[String](
    "--positive_train_background", positiveBackgroundFile,
    "--negative_train_background", negativeBackgroundFile,
    "--validation_background", validationBackgroundFile,
    "--knowledge_selector", knowledgeSelector,
    "--memory_updater", memoryUpdater,
    "--entailment_model", entailmentModel,
    "--num_memory_layers", numMemoryLayers.toString
  )

  val memoryNetworkInputs: Set[(String, Option[Step])] = baseInputs ++ Set(
    (positiveBackgroundFile, Some(positiveBackgroundSearcher)),
    (negativeBackgroundFile, Some(negativeBackgroundSearcher)),
    (validationBackgroundFile, Some(validationBackgroundSearcher))
  )
}

/**
 * This NeuralNetworkTrainer is a memory network, corresponding to memory_network.py.  We take good
 * and bad sentences as input, as well as a collection of background sentences for each input.
 */
class MemoryNetworkTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends MemoryNetworkBase(params, fileUtil) {
  override val name = "Neural Network Trainer, on sentences, with background knowledge input"

  val validParams = memoryNetworkParams
  JsonHelper.ensureNoExtras(params, name, validParams)

  override val arguments = Seq("MemoryNetworkSolver") ++ memoryNetworkArguments
  override val inputs = memoryNetworkInputs
  override val outputs = modelOutputs.toSet

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_params.json"
}

class DifferentiableSearchTrainer(
  params: JValue,
  fileUtil: FileUtil
) extends MemoryNetworkBase(params, fileUtil) {
  override val name = "DifferentiableSearchTrainer"

  val validParams = memoryNetworkParams ++ Seq(
    "corpus",
    "num epochs delay",
    "num epochs per encoding"
  )
  JsonHelper.ensureNoExtras(params, name, validParams)

  val corpus = (params \ "corpus").extract[String]
  val numEpochsDelay = JsonHelper.extractWithDefault(params, "num epochs delay", 10)
  val numEpochsPerEncoding = JsonHelper.extractWithDefault(params, "num epochs per encoding", 2)

  override val arguments = Seq[String](
    "DifferentiableSearchSolver",
    "--corpus", corpus,
    "--num_epochs_delay", numEpochsDelay.toString,
    "--num_epochs_per_encoding", numEpochsPerEncoding.toString
  ) ++ memoryNetworkArguments

  override val inputs: Set[(String, Option[Step])] = memoryNetworkInputs ++ Set(
    (corpus, None)
  )
  override val outputs = modelOutputs.toSet

  override val inProgressFile = modelPrefix + "_in_progress"
  override val paramFile = modelPrefix + "_params.json"
}
