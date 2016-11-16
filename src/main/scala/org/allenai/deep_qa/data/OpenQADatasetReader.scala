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

    val questions = jsonQuestions.map(item => {
      val question = (item \ "question").extract[String]
      val correctAnswer = (item \ "answer").extract[String]
      val answerOptions = (item \ "choices").extract[Seq[String]]
      val index = answerOptions.indexOf(correctAnswer)

      val instances = answerOptions.map(option =>{
        val statement = question + " " + option
        val truthValue = option == correctAnswer
        TrueFalseInstance(statement, Some(truthValue))

      })
      MultipleTrueFalseInstance(instances,  Some(index))
    })
    Dataset(questions)
    // TODO: Not correct.
  }
}
