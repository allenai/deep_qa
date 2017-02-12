package org.allenai.deep_qa.data

import scala.collection.mutable

import com.mattg.util.FileUtil
import org.json4s._
import org.json4s.Formats
import org.json4s.native.JsonMethods.parse

class SciQDatasetReader(fileUtil: FileUtil) extends DatasetReader[McQuestionAnswerInstance] {
  override def readFile(filename: String): Dataset[McQuestionAnswerInstance] = {
    val json = parse(fileUtil.readFileContents(filename))
    val instanceTuples = for {
      JObject(mcQuestion) <- json
      JField("support", JString(passage)) <- mcQuestion
      JField("question", JString(question)) <- mcQuestion
      JField("distractor1", JString(distractor1)) <- mcQuestion
      JField("distractor2", JString(distractor2)) <- mcQuestion
      JField("distractor3", JString(distractor3)) <- mcQuestion
      JField("correct_answer", JString(correctAnswer)) <- mcQuestion
    } yield (passage, question, distractor1, distractor2, distractor3, correctAnswer)

    val instances = instanceTuples.map { case (passage, question, distractor1,
      distractor2, distractor3, correct_answer) => {
      val options = Seq(distractor1, distractor2, distractor3, correct_answer).sorted
      val label = options.indexOf(correct_answer)
      McQuestionAnswerInstance(passage, question, options, Some(label))
    }}
    Dataset(instances)
  }
}
