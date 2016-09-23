package org.allenai.dlfa.data

import org.scalatest._

import com.mattg.util.FakeFileUtil

class ChildrensBookDatasetReaderSpec extends FlatSpecLike with Matchers {
  val fileUtil = new FakeFileUtil

  val datasetFile = "/dataset"
  val datasetFileContents = """1 sentence 1
  |2 sentence 2
  |3 sentence 3
  |4 sentence 4
  |5 sentence 5
  |6 sentence with XXXXX answer	the		in|of|the
  |
  |1 sentence 6
  |2 sentence 7
  |3 sentence 8
  |4 sentence 9
  |6 sentence has XXXXX response	no		a|no|neither
  |
  |""".stripMargin

  fileUtil.addFileToBeRead(datasetFile, datasetFileContents)

  val reader = new ChildrensBookDatasetReader(fileUtil)

  "readFile" should "return a correct dataset" in {
    val dataset = reader.readFile(datasetFile)
    dataset should be(Dataset(Seq(
      BackgroundInstance(
        QuestionAnswerInstance("sentence with XXXXX answer", Seq("in", "of", "the"), 2),
        Seq(
          "sentence 1",
          "sentence 2",
          "sentence 3",
          "sentence 4",
          "sentence 5"
        )
      ),
      BackgroundInstance(
        QuestionAnswerInstance("sentence has XXXXX response", Seq("a", "no", "neither"), 1),
        Seq(
          "sentence 6",
          "sentence 7",
          "sentence 8",
          "sentence 9"
        )
      )
    )))
  }
}
