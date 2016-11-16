package org.allenai.deep_qa.data

import org.scalatest._

import com.mattg.util.FakeFileUtil

class DatasetSpec extends FlatSpecLike with Matchers {
  val fileUtil = new FakeFileUtil

  "writeToFiles" should "write a single file correctly without indices" in {
    val filename = "/dataset"
    val contents = "sentence with XXXXX answer\tin###of###the\t2\n" +
      "sentence has XXXXX response\ta###no###neither\t1\n"
    fileUtil.addExpectedFileWritten(filename, contents)
    val dataset = Dataset(Seq(
      QuestionAnswerInstance("sentence with XXXXX answer", Seq("in", "of", "the"), Some(Seq(2))),
      QuestionAnswerInstance("sentence has XXXXX response", Seq("a", "no", "neither"), Some(Seq(1)))
    ))
    dataset.writeToFiles(Seq(filename), false, fileUtil)
    fileUtil.expectFilesWritten()
  }

  "writeToFiles" should "write a single file correctly with indices" in {
    val filename = "/dataset"
    val contents = "0\tsentence with XXXXX answer\tin###of###the\t2\n" +
      "1\tsentence has XXXXX response\ta###no###neither\t1\n"
    fileUtil.addExpectedFileWritten(filename, contents)
    val dataset = Dataset(Seq(
      QuestionAnswerInstance("sentence with XXXXX answer", Seq("in", "of", "the"), Some(Seq(2))),
      QuestionAnswerInstance("sentence has XXXXX response", Seq("a", "no", "neither"), Some(Seq(1)))
    ))
    dataset.writeToFiles(Seq(filename), true, fileUtil)
    fileUtil.expectFilesWritten()
  }

  "writeToFiles" should "write background instances correctly with indices" in {
    val filename1 = "/dataset"
    val contents1 = "0\tsentence with XXXXX answer\tin###of###the\t2\n" +
      "1\tsentence has XXXXX response\ta###no###neither\t1\n"
    fileUtil.addExpectedFileWritten(filename1, contents1)
    val filename2 = "/background_dataset"
    val contents2 = "0\tb1\tb2\n1\tb3\tb4\n"
    fileUtil.addExpectedFileWritten(filename2, contents2)
    val dataset = Dataset(Seq(
      BackgroundInstance(
        QuestionAnswerInstance("sentence with XXXXX answer", Seq("in", "of", "the"), Some(Seq(2))),
        Seq("b1", "b2")
      ),
      BackgroundInstance(
        QuestionAnswerInstance("sentence has XXXXX response", Seq("a", "no", "neither"), Some(Seq(1))),
        Seq("b3", "b4")
      )
    ))
    dataset.writeToFiles(Seq(filename1, filename2), true, fileUtil)
    fileUtil.expectFilesWritten()
  }

  "writeToFiles" should "write multiple true/false instances correctly with indices" in {
    val filename1 = "/dataset"
    val contents1 = "0\tsentence with an answer\t1\n" +
      "1\tsentence with of answer\t0\n" +
      "2\tsentence with in answer\t0\n" +
      "3\thaving an incorrect response\t0\n" +
      "4\thaving the perfect response\t1\n" +
      "5\thaving precisely response\t0\n"
    fileUtil.addExpectedFileWritten(filename1, contents1)
    val filename2 = "/background_dataset"
    val contents2 = "0\tb1\n1\tb2\n2\tb3\n3\tb4\n4\tb5\n5\tb6\n"
    fileUtil.addExpectedFileWritten(filename2, contents2)
    val dataset = Dataset(Seq(
      MultipleTrueFalseInstance(Seq(
          BackgroundInstance(TrueFalseInstance("sentence with an answer", Some(true)), Seq("b1")),
          BackgroundInstance(TrueFalseInstance("sentence with of answer", Some(false)), Seq("b2")),
          BackgroundInstance(TrueFalseInstance("sentence with in answer", Some(false)), Seq("b3"))
      ), Some(0)),
      MultipleTrueFalseInstance(Seq(
        BackgroundInstance(TrueFalseInstance("having an incorrect response", Some(false)), Seq("b4")),
        BackgroundInstance(TrueFalseInstance("having the perfect response", Some(true)), Seq("b5")),
        BackgroundInstance(TrueFalseInstance("having precisely response", Some(false)), Seq("b6"))
      ), Some(1))
    ))
    dataset.writeToFiles(Seq(filename1, filename2), true, fileUtil)
    fileUtil.expectFilesWritten()
  }
}
