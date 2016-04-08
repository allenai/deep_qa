package org.allenai.semparse.pipeline.science_data

import org.json4s._
import org.json4s.JsonDSL._

object ScienceQuestionPipeline {
  val sentenceProcesserParams: JValue = ("max word count per sentence" -> 10) ~ ("output format" -> "training data")

  val kbGeneratorParams: JValue = ("sentences" -> sentenceProcesserParams)

  val questionProcesserParams: JValue = JNothing

  def main(args: Array[String]) {
    new KbGenerator(kbGeneratorParams).runPipeline()
    //new ScienceQuestionProcessor(questionProcesserParams).runPipeline()
  }
}
