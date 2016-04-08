package org.allenai.semparse.pipeline.base

import collection.mutable

import edu.cmu.ml.rtw.pra.experiments.ExperimentScorer

import com.mattg.util.FileUtil
import com.mattg.pipeline.MetricOutputter
import com.mattg.pipeline.Step

import org.json4s._
import org.json4s.native.JsonMethods.parse

class Evaluator(
  methods: Seq[(String, JValue)],
  fileUtil: FileUtil = new FileUtil
) extends Step(None, fileUtil) {
  implicit val formats = DefaultFormats

  val testers = methods.map(method => new Tester(method._2, fileUtil))

  override def name = "Evaluator"
  override def inputs = testers.map(t => (t.outputFile, Some(t))).toSet
  override def outputs = Set()

  val cacheFile = "results/cached_results.tsv"

  val outputFiles = testers.map(_.outputFile)
  val testQueryFile = {
    val files = testers.map(_.queryFile).toSet
    if (files.size != 1) {
      throw new IllegalStateException("All methods in an evaluation must use the same test file!")
    }
    files.head
  }
  val outputFileMap = methods.zip(outputFiles).map(p => (p._1._1, p._2)).toMap

  lazy val queryAnswers: Seq[(String, (Set[String], Set[String]))] = {
    val wholeJson = parse(fileUtil.readLinesFromFile(testQueryFile).mkString("\n"))
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
    }).toSeq
  }

  def getMethodResponses(method: String): Seq[Seq[(Double, String)]] = {
    val resultsFile = outputFileMap(method)
    val queryResponses = new mutable.ListBuffer[Seq[(Double, String)]]()
    var sentence = ""
    var query = ""
    var predictions = new mutable.ListBuffer[(Double, String)]
    for (line <- fileUtil.readLinesFromFile(resultsFile)) {
      if (line.isEmpty()) {
        if (!query.isEmpty()) {
          queryResponses += predictions.toSeq
        }
        sentence = ""
        query = ""
        predictions = new mutable.ListBuffer[(Double, String)]
      }
      if (sentence.isEmpty()) {
        sentence = line
      } else if (query.isEmpty()) {
        query = line
      } else {
        val fields = line.split(" ")
        val score = fields(0).toDouble
        val entity = fields(1)
        predictions += Tuple2(score, entity)
      }
    }
    queryResponses.toSeq
  }

  def taskComputer() = queryAnswers.map(q => (q._1, q._2._1))

  override def _runStep() {
    val methodOutputs = methods.map(_._1).zip(outputFiles)
    val methodTimestamps = methodOutputs.map(m => (m._1, fileUtil.getTimestamp(m._2)))
    val outputter = new MetricOutputter(
      methodTimestamps,
      getMethodResponses,
      taskComputer,
      cacheFile
    )
    outputter.scoreMethods()
  }
}
