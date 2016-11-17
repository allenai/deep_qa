package org.allenai.deep_qa.data

import com.mattg.util.FileUtil
import org.json4s.DefaultFormats
import org.json4s.native.JsonMethods

/**
 * Reader for the json pre-processed OpenQA dataset: https://github.com/uberspot/OpenTriviaQA.
  * Download and run the packaged ruby preprocessing step (ruby converter.rb /path/to/data\*)
 */
class OpenQADatasetReader(fileUtil: FileUtil) extends DatasetReader[MultipleTrueFalseInstance[TrueFalseInstance]] {
  override def readFile(filename: String): Dataset[MultipleTrueFalseInstance[TrueFalseInstance]] = {
    implicit val formats = DefaultFormats
    val source = fileUtil.readFileContents(filename)
    val jsonQuestions = JsonMethods.parse(source).children

    val questions = jsonQuestions.flatMap(item => {
      val question = (item \ "question").extract[String]
      val correctAnswer = (item \ "answer").extract[String]
      val answerOptions = (item \ "choices").extract[Seq[String]]
      val correctAnswerIndex = answerOptions.indexOf(correctAnswer)

      val assertOneAnswer = answerOptions.map(_.equals(correctAnswer)).count(_.equals(true)).equals(1)
      val instances = answerOptions.map(option =>{
        val statement = question + " " + option
        val truthValue = answerOptions.indexOf(option).equals(correctAnswerIndex)
        TrueFalseInstance(statement, Some(truthValue))
      })
      MultipleTrueFalseInstance[TrueFalseInstance](instances,  Some(correctAnswerIndex))
      if (assertOneAnswer) {
          Seq(MultipleTrueFalseInstance[TrueFalseInstance](instances, Some(correctAnswerIndex)))
        } else {
          Seq()
      }
    })
    Dataset(questions.filter(x => x.instances.length == 4))
  }
}
