package org.allenai.deep_qa.data

import scala.collection.mutable

import com.mattg.util.FileUtil

class ChildrensBookDatasetReader(
  fileUtil: FileUtil
) extends DatasetReader[BackgroundInstance[QuestionAnswerInstance]] {
  override def readFile(filename: String): Dataset[BackgroundInstance[QuestionAnswerInstance]] = {
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
