package org.allenai.semparse.evaluation

import java.io.FileWriter
import java.io.File

import scala.collection.mutable

import org.allenai.semparse.lisp.Environment
import org.allenai.semparse.training.Trainer

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import org.json4s._
import org.json4s.native.JsonMethods.parse

class Evaluator(
  env: Environment,
  queryFile: String,
  entityNames: => Map[String, Set[String]],
  fileUtil: FileUtil = new FileUtil
) {
  implicit val formats = DefaultFormats

  val catQueryFormat = "(expression-eval (quote (print-predicate-marginals %s (get-all-related-entities (list %s)))))"
  val relQueryFormat = "(expression-eval (quote (print-relation-marginals %s entity-tuple-array)))"
  val poolDepth = 100
  val runDepth = 200

  def evaluate(writer: FileWriter) {
    for (line <- fileUtil.getLineIterator(queryFile)) {
      println(line)
      val json = parse(line)
      val sentence = (json \ "sentence").extract[String]
      val queries = (json \ "queries").extract[Seq[JValue]]
      for (query <- queries) {
        val queryExpression = (query \ "queryExpression").extract[String]
        val midsInQuery = (query \ "midsInQuery").extract[Seq[String]]
        val correctIds = (query \ "correctAnswerIds").extract[Seq[String]]
        val numAnnotatedTrueEntities = correctIds.size
        val incorrectIds = (query \ "incorrectAnswerIds") match {
          case JArray(values) => values.map(_.extract[String]).toSet
          case _ => Set[String]()
        }
        val numAnnotatedFalseEntities = incorrectIds.size
        val isRelationQuery = (query \ "isRelationQuery") match {
          case JInt(i) => i == 1
          case _ => false
        }

        writer.write("\n")
        writer.write(sentence)
        writer.write("\n")
        writer.write(queryExpression)
        writer.write("\n")
        val expressionToEvaluate = if (isRelationQuery) {
          relQueryFormat.format(queryExpression)
        } else {
          val midStr = midsInQuery.map(m => "\"" + m + "\"").mkString(" ")
          catQueryFormat.format(queryExpression, midStr)
        }
        val result = env.evalulateSExpression(expressionToEvaluate)
        println(result)
      }
    }
  }
}

object Evaluator {
  def loadEntityNames(dataFile: String, fileUtil: FileUtil = new FileUtil) = {
    println(s"Loading entity names from file $dataFile ...")
    val midNames = new mutable.HashMap[String, mutable.HashSet[String]]
    for (line <- fileUtil.getLineIterator(dataFile)) {
      val fields = line.split("\t")
      val mids = fields(0).split(" ")
      val names = fields(1).trim().split("\" \"")
      for ((mid, name) <- mids.zip(names)) {
        midNames.getOrElseUpdate(mid, new mutable.HashSet[String]).add(name.replace("\"", ""))
      }
    }
    println("Done reading entity name file")
    midNames.par.mapValues(_.toSet).seq.toMap
  }

  def createEvaluationEnvironment(
    data: String,
    modelType: String,
    ranking: String,
    usingGraphs: Boolean
  ): Environment = {
    val dataFiles = data match {
      case "large" => Environment.COMMON_LARGE_DATA
      case "small" => Environment.COMMON_SMALL_DATA
      case other => throw new RuntimeException("unrecognized data option")
    }

    val modelFiles = modelType match {
      case "baseline" => Seq(Environment.BASELINE_MODEL_FILE)
      case "ensemble" => usingGraphs match {
        case true => Seq(Environment.BASELINE_MODEL_FILE, Environment.GRAPH_MODEL_FILE)
        case false => Seq(Environment.BASELINE_MODEL_FILE, Environment.DISTRIBUTIONAL_MODEL_FILE)
      }
      case "uschema" => usingGraphs match {
        case true => Seq(Environment.GRAPH_MODEL_FILE)
        case false => Seq(Environment.DISTRIBUTIONAL_MODEL_FILE)
      }
      case other => throw new RuntimeException("unrecognized model type")
    }

    val evalFile = modelType match {
      case "baseline" => Seq(Environment.EVAL_BASELINE_FILE)
      case "uschema" => Seq(Environment.EVAL_USCHEMA_FILE)
      case "ensemble" => Seq(Environment.EVAL_ENSEMBLE_FILE)
      case other => throw new RuntimeException("unrecognized model type")
    }

    val sfeSpecFile = data match {
      case "large" => Environment.LARGE_SFE_SPEC_FILE
      case "small" => Environment.SMALL_SFE_SPEC_FILE
      case other => throw new RuntimeException("unrecognized data option")
    }

    val serializedModelFile = Trainer.getModelFile(data, ranking, usingGraphs, modelType == "baseline")

    val baseInputFiles = dataFiles ++ Environment.ENV_FILES ++ modelFiles ++ evalFile
    val baseExtraArgs = Seq(sfeSpecFile)

    val (inputFiles, extraArgs) = modelType match {
      case "baseline" => (serializedModelFile +: baseInputFiles, baseExtraArgs)
      case _ => (baseInputFiles, baseExtraArgs :+ serializedModelFile)
    }

    new Environment(inputFiles, extraArgs, true)
  }

  def getOutputFile(
    data: String,
    modelType: String,
    ranking: String,
    usingGraphs: Boolean
  ) = {
    val graphs = if (usingGraphs) "with_graph_features" else "no_graph_features"
    modelType match {
      case "baseline" => s"results/${data}/baseline/results.txt"
      case other => s"results/${data}/${modelType}/${graphs}/${ranking}/results.txt"
    }
  }

  def filterInvalidConfigs(configs: Seq[(String, String, String, Boolean)]) = {
  }

  def main(args: Array[String]) {
    val fileUtil = new FileUtil

    val largeDataFile = "data/tacl2015-training.txt"
    val smallDataFile = "data/tacl2015-training-sample.txt"
    lazy val smallEntityNames = loadEntityNames(smallDataFile)
    lazy val largeEntityNames = loadEntityNames(largeDataFile)

    val queryFile = "data/tacl2015-test.txt"

    // What things should we evaluate?
    val datasets = Seq("small") //, "large")
    val modelTypes = Seq("uschema") //, "ensemble")
    val rankings = Seq("predicate") //, "query")
    val withGraphOrNot = Seq(true) //, false)
    val baselineConfigs = Seq(
      ("small", "baseline", "ignored", false)
      //("large", "baseline", "ignored", false)
    )

    // Generate all possible combinations
    val configs = for (data <- datasets;
         modelType <- modelTypes;
         ranking <- rankings;
         usingGraphs <- withGraphOrNot) yield (data, modelType, ranking, usingGraphs)

    // Now run the evaluation for them, assuming the model has been trained, and the evaluation
    // hasn't already been done.
    //(configs ++ baselineConfigs).par.foreach(config => {
    baselineConfigs.par.foreach(config => {
      val (data, modelType, ranking, usingGraphs) = config
      val modelFile = Trainer.getModelFile(data, ranking, usingGraphs, modelType == "baseline")
      val outputFile = getOutputFile(data, modelType, ranking, usingGraphs)
      if (!fileUtil.fileExists(modelFile)) {
        println(s"Model not available for configuration $config.  Skipping...")
      } else if (fileUtil.fileExists(outputFile)) {
        println(s"Evaluation already done for configuration $config.  Skipping...")
      } else {
        println(s"Running evaluation for $config")
        println("Creating environment")
        val env = createEvaluationEnvironment(data, modelType, ranking, usingGraphs)
        println("Getting entity names")
        lazy val entityNames = data match {
          case "large" => largeEntityNames
          case "small" => smallEntityNames
        }
        println("Creating evaluator")
        val evaluator = new Evaluator(env, queryFile, entityNames)
        fileUtil.mkdirs(new File(outputFile).getParent())
        val writer = fileUtil.getFileWriter(outputFile)
        evaluator.evaluate(writer)
        writer.close()
      }
    })
  }
}
