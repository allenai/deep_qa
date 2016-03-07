package org.allenai.semparse

import java.io.FileWriter
import java.io.File

import scala.collection.mutable

import com.jayantkrish.jklol.lisp.ConsValue

import edu.cmu.ml.rtw.users.matt.util.FileUtil

import org.json4s._
import org.json4s.native.JsonMethods.parse

// This class takes the output from Tester (which performs inference over a test set), and a set of
// annotated results, and performs an evaluation.
class Evaluator(
  resultsFile: String,
  queryFile: String,
  fileUtil: FileUtil = new FileUtil
) {
  implicit val formats = DefaultFormats

  def getQueryAnswers(): Map[String, (Set[String], Set[String])] = {
    val wholeJson = parse(fileUtil.readLinesFromFile(queryFile).mkString("\n"))
    wholeJson.extract[Seq[JValue]].flatMap(json => {
      val sentence = (json \ "sentence").extract[String]
      println(s"Scoring queries for sentence: $sentence")
      val queries = (json \ "queries").extract[Seq[JValue]]
      queries.map(query => {
        val queryExpression = (query \ "queryExpression").extract[String]
        val correctIds = (query \ "correctAnswerIds").extract[Seq[String]].toSet
        val incorrectIds = (query \ "incorrectAnswerIds") match {
          case JArray(values) => values.map(_.extract[String]).toSet
          case _ => Set[String]()
        }
        (queryExpression -> (correctIds.toSet, incorrectIds.toSet))
      })
    }).toMap
  }

  def evaluate(writer: FileWriter) {
    val queryAnswers = getQueryAnswers()

    val queryResponses = new mutable.ListBuffer[(String, String, Seq[String])]()
    var sentence = ""
    var query = ""
    var predictions: List[String] = Nil
    for (line <- fileUtil.readLinesFromFile(resultsFile)) {
      if (line.isEmpty()) {
        if (!query.isEmpty()) {
          queryResponses += Tuple3(sentence, query, predictions)
        }
        sentence = ""
        query = ""
        predictions = Nil
      }
      if (sentence.isEmpty()) {
        sentence = line
      } else if (query.isEmpty()) {
        query = line
      } else {
        predictions = line :: predictions
      }
    }

    val numQueries = queryResponses.size
    var averagePrecisionSum = 0.0
    var weightedAPSum = 0.0
    var numNoAnswerQueries = 0
    var numAnnotatedTrueEntities = 0
    var numAnnotatedFalseEntities = 0
    val pointPrecisionSum = new Array[Double](11)

    for ((sentence, query, responses) <- queryResponses) {
      // I think I want this to crash if there's a mismatch, so we'll go with this for now.
      val (correctIds, incorrectIds) = queryAnswers(query)

      writer.write(sentence)
      writer.write("\n")
      writer.write(query)
      writer.write("\n")
      for (response <- responses) {
        val entity = response.split(" ")(1)
        if (correctIds.contains(entity)) {
          writer.write("1 ")
        } else if (incorrectIds.contains(entity)) {
          writer.write("0 ")
        } else {
          writer.write("? ")
        }
        writer.write(response)
        writer.write("\n")
      }
      val entityScores = responses.map(line => {
        val fields = line.split(" ")
        val score = fields(0).toDouble
        val entity = fields(1)
        (score, entity)
      })
      val averagePrecision = Evaluator.computeAveragePrecision(entityScores, correctIds)
      averagePrecisionSum += averagePrecision
      writer.write(s"AP: $averagePrecision\n")
      val weightedAveragePrecision = averagePrecision * correctIds.size
      weightedAPSum += weightedAveragePrecision
      writer.write(s"WAP: $weightedAveragePrecision\n")

      val pointPrecision = Evaluator.compute11PointPrecision(entityScores, correctIds)
      writer.write("11-point precision/recall: [")
      for (i <- 0 until pointPrecision.length) {
        if (i != 0) {
          writer.write(", ")
        }
        writer.write(pointPrecision(i).toString)
        pointPrecisionSum(i) += pointPrecision(i)
      }
      writer.write("]\n\n")
    }

    // Now some global statistics.
    val meanAveragePrecision = averagePrecisionSum / numQueries
    val weightedMAP = weightedAPSum / numAnnotatedTrueEntities
    val reweightedMAP = averagePrecisionSum / (numQueries - numNoAnswerQueries)
    writer.write(s"MAP: $meanAveragePrecision\n")
    writer.write(s"Weighted MAP: $weightedMAP\n")
    writer.write(s"Reweighted MAP: $reweightedMAP\n")

    writer.write("11-point averaged precision/recall: [")
    for (i <- 0 until pointPrecisionSum.length) {
      if (i != 0) {
        writer.write(", ")
      }
      writer.write((pointPrecisionSum(i) / numQueries).toString)
    }
    writer.write("]\n")

    writer.write("---\n")
    writer.write(s"Num queries: $numQueries\n")
    writer.write(s"annotated true answers: $numAnnotatedTrueEntities\n")
    writer.write(s"annotated false answers: $numAnnotatedFalseEntities\n")
    writer.write(s"Queries with no possible answers: $numNoAnswerQueries\n")
  }
}

object Evaluator {

  // A helper method that really could belong in a different class.  Just computing average
  // precision from a list of scored objects.  We group and sort this list, just in case it wasn't
  // done previously.  I don't expect that the inputs will ever be long enough that that's a
  // problem.
  def computeAveragePrecision[T](scores: Seq[(Double, T)], correctInstances: Set[T]): Double = {
    if (correctInstances.size == 0) return 0
    var numPredictionsSoFar = 0
    var totalPrecision = 0.0
    var correctSoFar = 0.0  // this is a double to avoid casting later

    // These are double scores, so ties should be uncommon, but they do happen.
    val grouped = scores.groupBy(_._1).toSeq.sortBy(-_._1)
    for (resultsWithScore <- grouped) {
      val score = resultsWithScore._1
      for ((score, instance) <- resultsWithScore._2) {
        numPredictionsSoFar += 1
        if (correctInstances.contains(instance)) {
          correctSoFar += 1.0
        }
      }
      for ((score, instance) <- resultsWithScore._2) {
        if (correctInstances.contains(instance)) {
          totalPrecision += (correctSoFar / numPredictionsSoFar)
        }
      }
    }
    totalPrecision / correctInstances.size
  }

  // Another helper method that could belong in a different class.  Here's we're computing a
  // simplified precision recall curve with just 11 points (though, really, we could output the
  // whole list and get a finer-grained curve...).  The returned Seq[Double] will have 11 entries.
  def compute11PointPrecision[T](scores: Seq[(Double, T)], correctInstances: Set[T]): Seq[Double] = {
    var correctSoFar = 0.0  // this is a double to avoid casting later
    val precisionRecall = scores.zipWithIndex.map(instanceScoreIndex => {
      val ((score, instance), index) = instanceScoreIndex
      if (correctInstances.contains(instance)) {
        correctSoFar += 1.0
      }
      val precision = correctSoFar / (index + 1)
      val recall = if (correctInstances.size == 0) 0 else correctSoFar / correctInstances.size
      (precision, recall)
    })

    var maxPrecision = 0.0
    val interpolatedPrecision = precisionRecall.map(_._1).toArray
    for (i <- 0 until precisionRecall.size) {
      val index = precisionRecall.size - (i + 1)
      maxPrecision = Math.max(maxPrecision, precisionRecall(index)._1)
      interpolatedPrecision(index) = maxPrecision
    }

    (0 to 10).map(i => {
      val point = i * .1
      val recallIndex = precisionRecall.indexWhere(_._2 >= point)
      if (recallIndex >= 0) interpolatedPrecision(recallIndex) else 0.0
    })
  }

  // To make the results file more human-readable, we show entity names along with MIDs.  This just
  // reads a mapping from mid to entity names from the input training file.  And it's in the
  // Evaluator object instead of the class because we really just need to do this once, and all
  // Evaluator objects can reuse this map.
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
      case "baseline" => s"results/${data}_acl2016/baseline/annotated_output.txt"
      case other => s"results/${data}_acl2016/${modelType}/${ranking}/${ensemble}/annotated_output.txt"
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
      val resultsFile = Tester.getOutputFile(data, ranking, modelType, ensembledEvaluation)
      val annotatedFile = getOutputFile(data, modelType, ranking, ensembledEvaluation)
      if (!fileUtil.fileExists(resultsFile)) {
        println(s"Results not available for configuration $config.  Skipping...")
      } else if (fileUtil.fileExists(annotatedFile)) {
        println(s"Evaluation already done for configuration $config.  Skipping...")
      } else {
        println(s"Running evaluation for $config")
        println("Creating environment")
        println("Creating evaluator")
        val evaluator = new Evaluator(resultsFile, Experiments.TEST_DATA_FILE)
        fileUtil.mkdirs(new File(annotatedFile).getParent())
        val writer = fileUtil.getFileWriter(annotatedFile)
        evaluator.evaluate(writer)
        writer.close()
      }
    })
  }
}
