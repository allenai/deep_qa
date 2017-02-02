package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import org.json4s._

import org.allenai.deep_qa.data.DatasetReader

import scala.collection.mutable.Map
import scala.sys.process.Process
import scala.sys.process.ProcessLogger

/**
  * This Step is a SentenceProducer that combines a QuestionAnswerInstance
  * with background sentences from a BackgroundCorpusSearcher.
  * The QuestionAnswerInstance input file has the format
  * "[index][tab][question][tab][options][tab][label]", and the file with
  * background sentences has the format
  * "[index][tab][background1][tab][background2][tab]...". The output file
  * is in the format of a MCReadingComprehensionInstance with
  * "[index][tab][passage][tab][question][tab][options][tab][label]".
  * The background sentences compose the passage in this case.
 */
class QaBackgroundToMcStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "QA Background To MC Step"

  val validParams = baseParams ++ Seq("sentences", "background", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val sentenceProducer = SentenceProducer.create(params \ "sentences", fileUtil)
  val sentencesFile = sentenceProducer.outputFile

  val backgroundProducer = SentenceProducer.create(params \ "background", fileUtil)
  val backgroundFile = backgroundProducer.outputFile

  override val outputFile = JsonHelper.extractAsOption[String](params, "output file") match {
    case None => sentencesFile.dropRight(4) + "_with_background_as_passages.tsv"
    case Some(filename) => filename
  }

  override val inputs: Set[(String, Option[Step])] = Set(
    (sentencesFile, Some(sentenceProducer)),
    (backgroundFile, Some(backgroundProducer))
  )
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.dropRight(4) + "_params.json"
  override val inProgressFile = outputs.head.dropRight(4) + "_in_progress"

  override def _runStep() {
    fileUtil.mkdirsForFile(outputFile)
    // make a map from indexes to paragraphs
    val passageMap = fileUtil.mapLinesFromFile(backgroundFile, line => {
      val fields = line.split("\t")
      val index = fields(0).toInt
      val passage = fields.drop(1).mkString(" ")
      (index, passage)
    }).toMap

    val outputLines = fileUtil.mapLinesFromFile(sentencesFile, line => {
      val fields = line.split("\t")
      val index = fields(0).toInt
      val sentence = fields.drop(1)
      // retrieve the passage corresponding to the index from the map
      val passage = passageMap.getOrElse(index, "")
      index.toString + "\t" + passage + "\t" + sentence.mkString("\t")
    })
    fileUtil.writeLinesToFile(outputFile, outputLines)
  }
}
