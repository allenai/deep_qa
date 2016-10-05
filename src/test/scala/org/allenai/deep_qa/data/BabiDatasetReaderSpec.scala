package org.allenai.deep_qa.data

import com.mattg.util.FakeFileUtil
import org.scalatest._

class BabiDatasetReaderSpec extends FlatSpecLike with Matchers {

  val fileUtil = new FakeFileUtil
  val datasetFile = "/dataset"
  val datasetFileContents = """1 background1
                              |2 background2
                              |3 background3
                              |4 background4
                              |5 background5
                              |6 question?	answer1	1,3
                              |1 background6
                              |2 background7
                              |3 background8
                              |4 background9
                              |5 multiple option question?	answer2,answer3	1,3,4
                              |
                              |""".stripMargin

  fileUtil.addFileToBeRead(datasetFile, datasetFileContents)
  val reader = new BabiDatasetReader(fileUtil)

  "getAnswerVocabulary" should "return all possible answers" in {

    val answerVocabulary = reader.getAnswerVocabulary(datasetFile).toSet

    answerVocabulary should be(Set("answer1", "answer2", "answer3"))
  }

   "readFile" should "return a correct dataset" in {
     val dataset = reader.readFile(datasetFile)

    dataset should be(Dataset(Seq(
      BackgroundInstance(
        MultipleCorrectQAInstance("question?", Seq("answer1", "answer2", "answer3"), Some(Seq(0))),
        Seq(
          "background1",
          "background2",
          "background3",
          "background4",
          "background5"
        )
      ),
      BackgroundInstance(
        MultipleCorrectQAInstance("multiple option question?", Seq("answer1", "answer2", "answer3"), Some(Seq(1,2))),
        Seq(
          "background6",
          "background7",
          "background8",
          "background9"
        )
      )
    )))
  }


}
