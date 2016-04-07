package org.allenai.semparse.pipeline

import org.scalatest._

import org.json4s._

class ScienceQuestionProcessorSpec extends FlatSpecLike with Matchers {
  val processor = new ScienceQuestionProcessor(JNothing)
  "parseQuestionLine" should "correctly split the question and answer, and the answer options" in {
    val line = "B\tSentence 1. Sentence 2 ___. (A) answer 1 (B) answer 2 (C) answer 3 (D) answer 4"
    processor.parseQuestionLine(line) should be(ScienceQuestion(
      Seq("Sentence 1.", "Sentence 2 ___."),
      Seq(
        Answer("answer 1", false),
        Answer("answer 2", true),
        Answer("answer 3", false),
        Answer("answer 4", false)
      )
    ))
  }
}
