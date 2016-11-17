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
      MultipleTrueFalseInstance[TrueFalseInstance](
        List(TrueFalseInstance("Question1 ||| Answer1", Some(true)),
             TrueFalseInstance("Question1 ||| Answer2", Some(false)),
             TrueFalseInstance("Question1 ||| Answer3", Some(false)),
             TrueFalseInstance("Question1 ||| Answer4", Some(false))),Some(0)),

      MultipleTrueFalseInstance[TrueFalseInstance](
        List(TrueFalseInstance("Question2 ||| Answer1", Some(false)),
          TrueFalseInstance("Question2 ||| Answer2", Some(true)),
          TrueFalseInstance("Question2 ||| Answer3", Some(false)),
          TrueFalseInstance("Question2 ||| Answer4", Some(false))),Some(1)))
    ))
  }
}