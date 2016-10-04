package org.allenai.deep_qa.data

import scala.collection.mutable
import com.mattg.util.FileUtil

class BabiDatasetReader(fileUtil: FileUtil) {

    def readFile(filename: String): Dataset[BackgroundInstance[QuestionAnswerInstance]] = {
      val answerVocabulary = getAnswerVocabulary(filename)
      val currentBackground = new mutable.ArrayBuffer[String]
      val currentQuestions = new mutable.ArrayBuffer[String]
      val questions = new mutable.ArrayBuffer[BackgroundInstance[QuestionAnswerInstance]]

      for (line <- fileUtil.getLineIterator(filename).filter(x => !x.isEmpty)){
        val splitline = line.split("\t")

        // If the line index is 1, we start a new "story", so we
        // clear the sentences as they are no longer relevant.
        if (line.split(" ")(0).toInt == 1){
          currentBackground.clear()
        }

        if (splitline.length == 1) {
          val backgroundEntry = line.split(" ")
          currentBackground += backgroundEntry.drop(1).mkString(" ")
        }
        else {
          val thisQuestionSet = parseQuestion(line, currentBackground, answerVocabulary)
          questions.appendAll(thisQuestionSet)
        }

      }
      Dataset(questions.toList)
    }

  /**
    * In order to make all the different Babi questions formats fit into QuestionAnswerInstances, we first
    * iterate over the whole file and retrieve all the possible answers. This then forms the set of possible
    * answers for every BackgroundInstance.
    */

  def getAnswerVocabulary(filename:String): List[String] = {

    var allAnswers = mutable.Set[String]()

    for (line <- fileUtil.getLineIterator(filename)) {
      val splitLine = line.split("\t")
      if (splitLine.length != 1) {
        allAnswers ++= splitLine(1).split(",")

      }
    }
    allAnswers.toList
  }

  /**
    * This returns a list of BackgroundInstances as single questions in the Babi dataset can have multiple answers.
    * Eg: What was Matt holding?  eggs,football. In this case, we make Instances for both correct answers.
    */
  def parseQuestion(line: String,
                    currentBackground: mutable.ArrayBuffer[String],
                    answerVocabulary: List[String]): List[BackgroundInstance[QuestionAnswerInstance]] = {

    val splitFields = line.split("\t")
    val question = splitFields(0).split(" ").drop(1).mkString(" ")
    val allAnswers = splitFields(1).split(",")
    val thisBackground = currentBackground.toArray.toList

    allAnswers.map(x => BackgroundInstance(
      QuestionAnswerInstance(
        question, answerVocabulary, Some(answerVocabulary.indexOf(x))), thisBackground)).toList

  }
}

object BabiDatasetReader {
  def main(args: Array[String]) {
    val fileUtil = new FileUtil
    val reader = new BabiDatasetReader(fileUtil)
    val dataset = reader.readFile("/Users/markn/allen_ai/data/facebook/babi/en/qa2_two-supporting-facts_train.txt")
    dataset.writeToFiles(
      Seq("/Users/markn/allen_ai/data/facebook/train.tsv", "/Users/markn/allen_ai/data/facebook/train_background.tsv"),
      true,
      fileUtil
    )
  }
}