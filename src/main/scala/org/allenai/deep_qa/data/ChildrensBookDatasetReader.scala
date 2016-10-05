package org.allenai.deep_qa.data

import scala.collection.mutable

import com.mattg.util.FileUtil

class ChildrensBookDatasetReader(fileUtil: FileUtil) {
  def readFile(filename: String): Dataset[BackgroundInstance[QuestionAnswerInstance]] = {
    val currentLines = new mutable.ArrayBuffer[String]
    val questions = new mutable.ArrayBuffer[BackgroundInstance[QuestionAnswerInstance]]
    for (line <- fileUtil.getLineIterator(filename)) {
      if (line.isEmpty) {
        questions += convertLinesToQuestion(currentLines)
        currentLines.clear()
      } else {
        currentLines += line
      }
    }
    Dataset(questions)
  }

  def convertLinesToQuestion(lines: Seq[String]): BackgroundInstance[QuestionAnswerInstance] = {
    val questionLine = lines.last
    val backgroundLines = lines.dropRight(1)
    val background = backgroundLines.map(_.split(" ").drop(1).mkString(" "))
    val questionFields = questionLine.split("\t")
    val questionText = questionFields(0).split(" ").drop(1).mkString(" ")
    val answerString = questionFields(1)
    val answerOptionString = questionFields(3)
    val answerOptions = answerOptionString.split("\\|")
    val label = answerOptions.indexOf(answerString)
    BackgroundInstance(QuestionAnswerInstance(questionText, answerOptions, Some(Seq(label))), background)
  }
}

// TODO(matt): Move this somewhere else, or just remove it, once this is integrated as a Step in
// the pipeline.
object ChildrensBookDatasetReader {
  def main(args: Array[String]) {
    val fileUtil = new FileUtil
    val reader = new ChildrensBookDatasetReader(fileUtil)
    val dataset = reader.readFile("/home/mattg/data/facebook/childrens_books/data/cbtest_CN_train.txt")
    dataset.writeToFiles(
      Seq("/home/mattg/data/facebook/train.tsv", "/home/mattg/data/facebook/train_background.tsv"),
      true,
      fileUtil
    )
  }
}
