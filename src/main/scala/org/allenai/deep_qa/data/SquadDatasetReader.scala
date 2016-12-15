package org.allenai.deep_qa.data

import scala.collection.mutable

import com.mattg.util.FileUtil
import org.json4s._
import org.json4s.native.JsonMethods.parse

class SquadDatasetReader(fileUtil: FileUtil) extends DatasetReader[SquadInstance] {
  override def readFile(filename: String): Dataset[SquadInstance] = {
    val json = parse(fileUtil.readFileContents(filename))
    val instanceTuples = for {
      JObject(article) <- json \ "data"
      JField("paragraphs", JArray(paragraphs)) <- article
      JObject(paragraph_jval) <- paragraphs
      JField("context", JString(paragraph)) <- paragraph_jval
      JField("qas", JArray(questions)) <- paragraph_jval
      JObject(question_jval) <- questions
      JField("question", JString(question)) <- question_jval
      JField("answers", JArray(answers)) <- question_jval
    } yield (paragraph, question, answers)

    val instances = instanceTuples.map { case (paragraph, question, answers) => {
      // There are several annotations for each of the questions in the dev set, and they don't
      // always match.  This isn't really the right way to handle their dev set, as you really
      // should be using their script, but we'll do our best to handle that case here.  We'll just
      // take the majority vote for the answer (picking randomly on ties).
      val answerTuples = for {
        JObject(answer_jval) <- answers
        JField("answer_start", JInt(answerStart)) <- answer_jval
        JField("text", JString(answerText)) <- answer_jval
      } yield (answerStart.toInt, answerText)
      val uniqueAnswers = answerTuples.groupBy(identity).mapValues(_.size).toList.sortBy(-_._2)
      val mostFrequentAnswer = uniqueAnswers.head._1
      val (answerStart, answerText) = mostFrequentAnswer
      SquadInstance(question, paragraph, Some(answerStart), Some(answerStart + answerText.size))
    }}
    Dataset(instances)
  }
}
