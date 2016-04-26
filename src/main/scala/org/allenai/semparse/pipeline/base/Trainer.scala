package org.allenai.semparse.pipeline.base

import java.io.File

import org.allenai.semparse.Environment

import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import com.mattg.pipeline.Step

import org.json4s._

class Trainer(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats

  val validParams = Seq("model type", "ranking", "feature computer")
  JsonHelper.ensureNoExtras(params, "trainer", validParams)

  // The parameters we take.
  val ranking = JsonHelper.extractOptionWithDefault(params, "ranking", Seq("query", "predicate"), "query")
  val modelType = JsonHelper.extractOption(params, "model type", Seq("distributional", "formal", "combined"))

  // A few necessary things from previous steps.
  val pmiComputer = new SparkPmiComputer(params \ "feature computer", fileUtil)
  val processor = pmiComputer.featureComputer.trainingDataProcessor
  val dataName = processor.dataName

  // The output file.
  val serializedModelFile = s"output/$dataName/$modelType/$ranking/model.ser"

  // These are code files that are common to all models, and just specify the base lisp
  // environment.
  val baseEnvFile = "src/main/lisp/environment.lisp"
  val uschemaEnvFile = "src/main/lisp/uschema_environment.lisp"

  // These are code files, specifying the model we're using.
  val lispModelFile = s"src/main/lisp/model_${modelType}.lisp"
  val rankingLispFile = s"src/main/lisp/${ranking}_ranking.lisp"
  val trainingLispFile = "src/main/lisp/train_model.lisp"

  val handwrittenLispFiles =
    Seq(baseEnvFile, uschemaEnvFile, rankingLispFile, lispModelFile, trainingLispFile)

  // This one is a (hand-written) parameter file that will be passed to lisp code.
  val sfeSpecFile = pmiComputer.featureComputer.sfeSpecFile

  // These are data files, produced by TrainingDataProcessor.
  val entityFile = s"data/${dataName}/entities.lisp"
  val wordsFile = s"data/${dataName}/words.lisp"
  val jointEntityFile = s"data/${dataName}/joint_entities.lisp"
  val logicalFormFile = s"data/${dataName}/${ranking}_ranking_lf.lisp"

  val dataFiles = if (ranking == "query") {
    Seq(entityFile, wordsFile, jointEntityFile, logicalFormFile)
  } else {
    Seq(entityFile, wordsFile, logicalFormFile)
  }

  val featureFiles = Seq(
    s"data/${dataName}/mid_features.tsv",
    s"data/${dataName}/mid_pair_features.tsv",
    s"data/${dataName}/cat_word_features.tsv",
    s"data/${dataName}/rel_word_features.tsv"
  )

  // The path to the model file already encodes all of our parameters, really, but this will also
  // check that our data parameters haven't changed (i.e., using a different subset of the data,
  // but calling it by the same name).
  override val paramFile = serializedModelFile.replace("model.ser", "params.json")
  override val inProgressFile = serializedModelFile.replace("model.ser", "in_progress")
  override val name = "Model trainer"
  override val inputs =
    handwrittenLispFiles.map((_, None)).toSet ++
    Set((sfeSpecFile, None)) ++
    dataFiles.map((_, Some(processor))).toSet ++
    featureFiles.map((_, Some(pmiComputer))).toSet
  override val outputs = Set(
    serializedModelFile
  )

  override def _runStep() {
    val inputFiles = dataFiles ++ handwrittenLispFiles
    val extraArgs = Seq(sfeSpecFile, dataName, serializedModelFile)

    // Just creating the environment will train and output the model.
    new Environment(inputFiles, extraArgs)
  }
}
