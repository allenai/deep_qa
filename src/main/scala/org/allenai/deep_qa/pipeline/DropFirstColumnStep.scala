package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import org.json4s._

import org.allenai.deep_qa.data.DatasetReader

import scala.sys.process.Process
import scala.sys.process.ProcessLogger

/**
  * This Step is a SentenceProducer that reads a file and simply drops the first column.
  * Expected file format is "[passage][tab][question][tab][choices][tab][label]" or .
  * "[index][tab][passage][tab][question][tab][choices][tab][label]".
  * The output file format is "[question][tab][choices][tab][label]" or
  * "[index][tab][question][tab][choices][tab][label]".
  * This is used, for example, with McQuestionAnswerInstances
  * to removes the passages from them, turning it into a file with the format of a
  * QuestionAnswerInstance.
 */
class DropFirstColumnStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "No Passage MC"

  val validParams = baseParams ++ Seq("sentences", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProducer = SentenceProducer.create(params \ "sentences", fileUtil)
  val sentencesFile = sentenceProducer.outputFile
  override val indexSentences = sentenceProducer.indexSentences

  override val outputFile = JsonHelper.extractAsOption[String](params, "output file") match {
    case None => sentencesFile.dropRight(4) + "_no_passage.tsv"
    case Some(filename) => filename
  }

  override val inputs: Set[(String, Option[Step])] = Set((sentencesFile, Some(sentenceProducer)))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    fileUtil.mkdirsForFile(outputFile)
    val outputLines = fileUtil.mapLinesFromFile(sentencesFile, line => {
      val fields = line.split("\t")
      val (index, sentences) = if (fields(0).forall(_.isDigit)) {
        (Some(fields(0).toInt), fields.drop(1))
      } else {
        (None, fields)
      }
      val indexString = index.map(_.toString + "\t").getOrElse("")
      indexString + sentences.drop(1).mkString("\t")
    })
    fileUtil.writeLinesToFile(outputFile, outputLines)
  }
}
