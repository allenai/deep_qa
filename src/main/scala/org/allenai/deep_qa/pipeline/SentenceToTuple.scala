package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext

import org.allenai.deep_qa.parse.TupleExtractor

import java.io.File

import scala.concurrent._
import scala.util.{Success, Failure}
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global

/**
 * Here we convert sentences into a set of tuples, using a TupleExtractor.
 *
 * INPUTS: a file containing sentences, one or more sentences per line.  Expected file format is
 * "[sentence index][tab][sentence][tab][sentence]...", or just "[sentence][tab][sentence]...".
 *
 * OUTPUTS: a file containing the tuples from the sentence, one sentence per line.  Output file
 * format is the same as the input file, except sentences are replaced with a set of tuples.  If
 * you had more than one sentence per line, you will not necessarily be able to recover which
 * tuples came from which sentence.  The tuples are formatted as [tuple][tab][tuple]..., where each
 * tuple is [subject]###[predicate]###[object1]...  We keep the same index, if an index is
 * provided.  Optionally, we drop a line if the tuple generation generation failed for that line.
 */
class SentenceToTuple(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Sentence to Tuple"

  val validParams = baseParams ++ Seq("sentences", "tuple extractor", "output file", "drop errors")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val dropErrors = JsonHelper.extractWithDefault(params, "drop errors", true)

  val sentenceProducer = SentenceProducer.create(params \ "sentences", fileUtil)
  val sentencesFile = sentenceProducer.outputFile
  val tupleExtractor = TupleExtractor.create(params \ "tuple extractor")

  override val outputFile = JsonHelper.extractAsOption[String](params, "output file") match {
    case None => sentencesFile.dropRight(4) + "_as_tuples.tsv"
    case Some(filename) => filename
  }

  override val inputs: Set[(String, Option[Step])] = Set((sentencesFile, Some(sentenceProducer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    fileUtil.mkdirsForFile(outputFile)
    val outputLines = fileUtil.flatMapLinesFromFile(sentencesFile, line => {
      val fields = line.split("\t")
      val (index, sentences) = if (fields(0).forall(_.isDigit)) {
        (Some(fields(0).toInt), fields.drop(1))
      } else {
        (None, fields)
      }
      val indexString = index.map(_.toString + "\t").getOrElse("")
      val tuples = sentences.flatMap(tupleExtractor.extractTuples)
      if (tuples.length > 0) {
        Seq(indexString + tuples.map(_.asString("###")).mkString("\t"))
      } else {
        if (dropErrors) {
          Seq()
        } else {
          Seq(indexString)
        }
      }
    })
    fileUtil.writeLinesToFile(outputFile, outputLines)
  }
}
