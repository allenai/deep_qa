package org.allenai.semparse

import java.io.FileWriter
import java.io.File

import scala.collection.mutable

import com.jayantkrish.jklol.lisp.ConsValue

import com.mattg.util.FileUtil

import org.json4s._
import org.json4s.native.JsonMethods.parse

// This class performs inference over a test set for a particular experiment configuration, which
// is encapsulated in the Environment object given as a parameter.  The query file contains all of
// the test queries that will be evaluated against the environment, and the entity names is just a
// map from mids to a human-readable entity name, for easier human scanning of the result file.
// The file util is my object that abstracts away accessing the file system (and allows for easy
// faking of the file system, for much easier tests).
class Tester(
  env: Environment,
  queryFile: String,
  entityNames: => Map[String, String],
  fileUtil: FileUtil = new FileUtil
) {
  implicit val formats = DefaultFormats

  val catQueryFormat = "(expression-eval (quote (get-predicate-marginals %s (find-related-entities (list %s) (list %s)))))"

  // NOTE: the test data currently does not have any relation queries, so this does not get used.
  val relQueryFormat = "(expression-eval (quote (get-relation-marginals %s entity-tuple-array)))"
  val poolDepth = 100
  val runDepth = 200

  def evaluateModel(writer: FileWriter) {
    // The test file is a JSON object, which is itself a list of objects containing the query
    // information.
    val wholeJson = parse(fileUtil.readLinesFromFile(queryFile).mkString("\n"))
    for (json <- wholeJson.extract[Seq[JValue]]) {
      val sentence = (json \ "sentence").extract[String]
      println(s"Scoring queries for sentence: $sentence")
      val queries = (json \ "queries").extract[Seq[JValue]]
      for (query <- queries) {
        val queryExpression = (query \ "queryExpression").extract[String]
        println(s"Query: $queryExpression")
        val midsInQuery = (query \ "midsInQuery").extract[Seq[String]]
        val midRelationsInQuery = (query \ "midRelationsInQuery").extract[Seq[JValue]]
        val isRelationQuery = (query \ "isRelationQuery") match {
          case JInt(i) => i == 1
          case _ => false
        }
        val expressionToEvaluate = if (isRelationQuery) {
          // This doesn't happen in the current test data, and so I'm not sure this works right.
          // For example, I don't think (get-relation-marginals ...) is even defined.
          println("RELATION QUERY!")
          relQueryFormat.format(queryExpression)
        } else {
          val midStr = midsInQuery.map(m => "\"" + m + "\"").mkString(" ")
          val midRelationStr = midRelationsInQuery.map(jval => {
            val list = jval.extract[Seq[JValue]]
            val word = list(0).extract[String]
            val arg = list(1).extract[String]
            val isSource = list(2).extract[Boolean]
            val isSourceStr = if (isSource) "#t" else "#f"
            s"(list $word $arg $isSourceStr)"
          }).mkString(" ")
          catQueryFormat.format(queryExpression, midStr, midRelationStr)
        }

        // Now run the expression through the evaluation code and parse the result.
        println("Evaluating the expression")
        val result = env.evaluateSExpression(expressionToEvaluate).getValue()
        val entityScoreObjects = result.asInstanceOf[Array[Object]]
        println(s"Done evaluating, ranking ${entityScoreObjects.length} results")
        val entityScores = entityScoreObjects.map(entityScoreObject => {
          val cons = entityScoreObject.asInstanceOf[ConsValue]
          val list = ConsValue.consListToList(cons, classOf[Object])
          val score = list.get(0).asInstanceOf[Double].toDouble
          val entity = list.get(1).asInstanceOf[String]
          (score, entity)
        }).toSeq.sortBy(-_._1).take(poolDepth)

        // Finally, compute statistics (like average precision) and output to a result file.
        writer.write(sentence)
        writer.write("\n")
        writer.write(queryExpression)
        writer.write("\n")
        for ((score, entity) <- entityScores) {
          val names = entityNames.getOrElse(entity, "NO ENTITY NAME!")
          if (score >= 0.0) {
            writer.write(s"$score $entity $names\n")
          }
        }
        writer.write("\n")
      }
    }
  }
}

object Tester {

  // To make the results file more human-readable, we show entity names along with MIDs.  This just
  // reads a mapping from mid to entity names from the input training file.  And it's in the Tester
  // object instead of the class because we really just need to do this once, and all Tester
  // objects can reuse this map.
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
    midNames.par.mapValues(_.toSeq.sorted.map(n => "\"" + n + "\"").mkString(" ")).seq.toMap
  }

  // This is kind of messy.  The issue is that we have lots of different lisp files that need to be
  // loaded depending on exactly what we're trying to evaluate (i.e., Jayant's original model, my
  // model that uses graph-based features, what kind of ranking was used to train, etc.).  This
  // method takes the experiment configuration and puts together the right input files to give to
  // an Environment object that will then run the evaluation.
  def createEvaluationEnvironment(
    data: String,
    modelType: String,
    ranking: String,
    ensembledEvaluation: Boolean
  ): Environment = {
    val dataFiles = data match {
      case "large" => Experiments.COMMON_LARGE_DATA
      case "small" => Experiments.COMMON_SMALL_DATA
      case other => throw new RuntimeException("unrecognized data option")
    }

    val modelFiles = modelType match {
      case "baseline" => Seq(Experiments.BASELINE_MODEL_FILE)
      case "formal" => ensembledEvaluation match {
        case true => Seq(Experiments.BASELINE_MODEL_FILE, Experiments.FORMAL_MODEL_FILE)
        case false => Seq(Experiments.FORMAL_MODEL_FILE)
      }
      case "distributional" => ensembledEvaluation match {
        case true => Seq(Experiments.BASELINE_MODEL_FILE, Experiments.DISTRIBUTIONAL_MODEL_FILE)
        case false => Seq(Experiments.DISTRIBUTIONAL_MODEL_FILE)
      }
      case "combined" => ensembledEvaluation match {
        case true => Seq(Experiments.BASELINE_MODEL_FILE, Experiments.COMBINED_MODEL_FILE)
        case false => Seq(Experiments.COMBINED_MODEL_FILE)
      }
      case other => throw new RuntimeException("unrecognized model type")
    }

    val evalFile = modelType match {
      case "baseline" => Seq(Experiments.EVAL_BASELINE_FILE)
      case other => ensembledEvaluation match {
        case true => Seq(Experiments.EVAL_ENSEMBLE_FILE)
        case false => Seq(Experiments.EVAL_USCHEMA_FILE)
      }
    }

    val sfeSpecFile = data match {
      case "large" => Experiments.LARGE_SFE_SPEC_FILE
      case "small" => Experiments.SMALL_SFE_SPEC_FILE
      case other => throw new RuntimeException("unrecognized data option")
    }


    val baseInputFiles = dataFiles ++ Experiments.ENV_FILES ++ modelFiles ++ evalFile
    val baseExtraArgs = Seq(sfeSpecFile)

    // Most of the model lisp files assume there is a serialized parameters object passed in as the
    // first extra argument.  The baseline, instead, has a lisp file as its "serialized model", so
    // we have to handle these two cases differently.
    val (inputFiles, extraArgs) = modelType match {
      case "baseline" => {
        val baselineModelFile = Trainer.getModelFile(data, ranking, modelType)
        (baselineModelFile +: baseInputFiles, baseExtraArgs)
      }
      case other => ensembledEvaluation match {
        case true => {
          val serializedModelFile = Trainer.getModelFile(data, ranking, modelType)
          val baselineModelFile = Trainer.getModelFile(data, ranking, modelType)
          (baselineModelFile +: baseInputFiles, baseExtraArgs :+ serializedModelFile)
        }
        case false => {
          val serializedModelFile = Trainer.getModelFile(data, ranking, modelType)
          (baseInputFiles, baseExtraArgs :+ serializedModelFile)
        }
      }
    }

    new Environment(inputFiles, extraArgs, true)
  }

  // Given the experiment configuration, where should the output results file live?
  def getOutputFile(
    data: String,
    modelType: String,
    ranking: String,
    ensembledEvaluation: Boolean
  ) = {
    val ensemble = if (ensembledEvaluation) "ensemble" else "uschema"
    modelType match {
      // ACK!  I need to make this more general...  The dataset should not be just "large" and
      // "small"
      case "baseline" => s"results/${data}_acl2016/baseline/output.txt"
      case other => s"results/${data}_acl2016/${modelType}/${ranking}/${ensemble}/output.txt"
    }
  }

  // Here's our main entry point to all of the evaluation.  What this method does is set up all of
  // the experiment configurations, checks to see if the model has been trained and if the
  // evaluation has already been run, and evaluates in parallel all of the remaining
  // configurations.
  def main(args: Array[String]) {
    val fileUtil = new FileUtil

    // We only want to load these once, because (especially on the large data set) it could take a
    // while.  But do it lazily, in case we're not running an experiment config that needs them.
    lazy val smallEntityNames = loadEntityNames(Experiments.SMALL_BASE_DATA_FILE)
    lazy val largeEntityNames = loadEntityNames(Experiments.LARGE_BASE_DATA_FILE)

    // We run an evaluation for experiment listed in Experiments.experimentConfigs, if the model
    // file is available and the evaluation hasn't already been done.
    Experiments.experimentConfigs.par.foreach(config => {
      val (data, modelType, ranking, ensembledEvaluation) = config
      val modelFile = Trainer.getModelFile(data, ranking, modelType)
      val outputFile = getOutputFile(data, modelType, ranking, ensembledEvaluation)
      if (!fileUtil.fileExists(modelFile)) {
        println(s"Model not available for configuration $config.  Skipping...")
      } else if (fileUtil.fileExists(outputFile)) {
        println(s"Evaluation already done for configuration $config.  Skipping...")
      } else {
        println(s"Running evaluation for $config")
        println("Creating environment")
        val env = createEvaluationEnvironment(data, modelType, ranking, ensembledEvaluation)
        println("Getting entity names")
        lazy val entityNames = data match {
          case "large" => largeEntityNames
          case "small" => smallEntityNames
        }
        println("Creating tester")
        val tester = new Tester(env, Experiments.TEST_DATA_FILE, entityNames)
        fileUtil.mkdirs(new File(outputFile).getParent())
        val writer = fileUtil.getFileWriter(outputFile)
        tester.evaluateModel(writer)
        writer.close()
      }
    })
  }
}
