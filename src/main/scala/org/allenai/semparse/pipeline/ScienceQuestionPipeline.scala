package org.allenai.semparse.pipeline

import org.json4s._
import org.json4s.JsonDSL._

object ScienceQuestionPipeline {
  val params: JValue = ("sentences per word file" -> 100)

  def main(args: Array[String]) {
    new ScienceSentenceProcessor(params).runPipeline()
  }
}
