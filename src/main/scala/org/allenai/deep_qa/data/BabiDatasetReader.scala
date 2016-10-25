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
          questions.append(parseQuestion(line, currentBackground, answerVocabulary))
        }
      }
      Dataset(questions.toList)
    }

  /**
    * In order to make all the different Babi question formats fit into QuestionAnswerInstances, we first
    * iterate over the whole file and retrieve all the possible answers. This then forms the set of possible
    * answers for every BackgroundInstance.
    */
  def getAnswerVocabulary(filename:String): Seq[String] = {
    // Lines that have a tab in them have answers in the second column, comma separated.
    // Lines with no tab have no answers.
    fileUtil.flatMapLinesFromFile(filename, line => {
      val fields = line.split("\t")
      if (fields.length != 1) {
        fields(1).split(",").toSeq
      } else {
        Seq()
      }
    }).distinct
  }

  /**
    * This returns a list of BackgroundInstances as single questions in the Babi dataset can have multiple answers.
    * Eg: What was Matt holding?  eggs,football. In this case, we make Instances for both correct answers.
    */
  def parseQuestion(line: String,
                    currentBackground: mutable.ArrayBuffer[String],
                    answerVocabulary: Seq[String]): BackgroundInstance[QuestionAnswerInstance] = {

    val splitFields = line.split("\t")
    val question = splitFields(0).split(" ").drop(1).mkString(" ")
    val allAnswers = splitFields(1).split(",")
    val thisBackground = currentBackground.toArray.toList
    val indicies = allAnswers.map(x => answerVocabulary.indexOf(x)).toSeq
    BackgroundInstance(
      QuestionAnswerInstance(
        question,
        answerVocabulary,
        Some(indicies)),
      thisBackground)
  }
}

object BabiDatasetReader {
  def main(args: Array[String]) {
    val fileUtil = new FileUtil
    val reader = new BabiDatasetReader(fileUtil)
    val dataset = reader.readFile("/home/mattg/data/facebook/babi_v1.0/en/qa1_single-supporting-fact_train.txt")
    fileUtil.mkdirs("/home/mattg/data/facebook/babi_v1.0/processed/")
    dataset.writeToFiles(
      Seq(
        "/home/mattg/data/facebook/babi_v1.0/processed/task_1_train.tsv",
        "/home/mattg/data/facebook/babi_v1.0/processed/task_1_background.tsv"
      ),
      true,  // withIndices
      fileUtil
    )
  }
}
