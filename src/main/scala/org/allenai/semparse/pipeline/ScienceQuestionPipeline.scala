package org.allenai.semparse.pipeline

import org.json4s._
import org.json4s.JsonDSL._

object ScienceQuestionPipeline {
  val params: JValue = ("max word count per sentence" -> 10) ~ ("output format" -> "debug")

  def main(args: Array[String]) {
    new ScienceSentenceProcessor(params).runPipeline()
  }
}
