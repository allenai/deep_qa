package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper
import org.json4s._

import org.allenai.deep_qa.data.DatasetReader

/**
 * This Step is a SentenceProducer that reads some external dataset and converts it into a format
 * usable by our pipeline, by means of a DatasetReader.  This is for doing things like running
 * experiments on the bAbI dataset, or the Facebook Children's Book Test, or other similar things.
 */
class DatasetReaderStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(None, fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "DatasetReaderStep"
  val validParams = baseParams.filterNot(_ == "max sentences") ++
    Seq("input file", "output files", "reader")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val inputFile = (params \ "input file").extract[String]
  val outputFiles = (params \ "output files").extract[List[String]]
  val reader = DatasetReader.readers((params \ "reader").extract[String])(fileUtil)

  override val inputs: Set[(String, Option[Step])] = Set((inputFile, None))
  override val outputs = outputFiles.toSet
  override val inProgressFile = outputs.head.dropRight(4) + s"_in_progress"

  override def _runStep() {
    val dataset = reader.readFile(inputFile)
    dataset.writeToFiles(outputFiles, indexSentences, fileUtil)
  }
}
