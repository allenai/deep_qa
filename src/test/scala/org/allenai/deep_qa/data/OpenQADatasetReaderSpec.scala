package org.allenai.deep_qa.data

import com.mattg.util.FakeFileUtil
import org.scalatest._

class OpenQADatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FakeFileUtil
  val datasetFile = "/dataset"
  val datasetFileContents = """[
      |{
      |"question": "Question1",
      |"answer": "Answer1",
      |"choices": [
      |"Answer1",
      |"Answer2",
      |"Answer3",
      |"Answer4"
      |]
      |},
      |{
      |"question": "Question2",
      |"answer": "Answer2",
      |"choices": [
      |"Answer1",
      |"Answer2",
      |"Answer3",
      |"Answer4"
      |]
      |}
      |]""".stripMargin

  fileUtil.addFileToBeRead(datasetFile, datasetFileContents)
  val reader = new OpenQADatasetReader(fileUtil)

  "readFile" should "return a correct dataset" in {
    val dataset = reader.readFile(datasetFile)

    dataset should be(Dataset(Seq(
        QuestionAnswerInstance("Question1", Seq("Answer1", "Answer2", "Answer3", "Answer4"), Some(Seq(0))),
        QuestionAnswerInstance("Question2", Seq("Answer1", "Answer2", "Answer3", "Answer4"), Some(Seq(1))))
    ))
  }
}