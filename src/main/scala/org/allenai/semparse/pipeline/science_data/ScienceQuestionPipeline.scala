package org.allenai.semparse.pipeline.science_data

import org.json4s._
import org.json4s.JsonDSL._

object ScienceQuestionPipeline {
  val sentenceProcesserParams: JValue = ("max word count per sentence" -> 10) ~ ("output format" -> "debug")
  val questionProcesserParams: JValue = ("question file" -> "data/science_questions.tsv")

  def main(args: Array[String]) {
    //new ScienceSentenceProcessor(sentenceProcesserParams).runPipeline()
    new ScienceQuestionProcessor(questionProcesserParams).runPipeline()
  }
}
