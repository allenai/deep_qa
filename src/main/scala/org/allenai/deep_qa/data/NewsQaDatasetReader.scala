package org.allenai.deep_qa.data

import scala.collection.mutable
import scala.collection.JavaConverters._

import com.opencsv.CSVReader
import com.mattg.util.FileUtil
import java.io.StringReader
import org.json4s._
import org.json4s.native.JsonMethods.parse

class NewsQaDatasetReader(fileUtil: FileUtil) extends DatasetReader[SpanPredictionInstance] {
  override def readFile(filename: String): Dataset[SpanPredictionInstance] = {
    val reader = new StringReader(fileUtil.readFileContents(filename))
    // read the NewsQA data and discard "bad" questions and questions that
    // do not have an answer present in the passage.
    // val header = csv.readNext()
    val csv = new CSVReader(reader)
    val instanceTuples = for {
      line <- csv.readAll().asScala.tail
      questionText = line(0)
      label = line(1)
      stringAns = line(2)
      passageText = line(3)
    } yield (questionText, passageText, label)
    val instances = instanceTuples.map { case (questionText, passageText, label) => {
      val labelSplit = label.split(":")
      // start character of answer in passage, inclusive
      val answerStart = labelSplit(0).toInt
      // end character of answer in passage, exclusive
      val answerEnd = labelSplit(1).toInt
      SpanPredictionInstance(questionText, passageText, Some(answerStart, answerEnd))
    }}
    Dataset(instances)
  }
}
