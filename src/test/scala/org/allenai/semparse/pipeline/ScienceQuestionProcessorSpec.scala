package org.allenai.semparse.pipeline

import org.scalatest._

import org.json4s._

class ScienceQuestionProcessorSpec extends FlatSpecLike with Matchers {
  val processor = new ScienceQuestionProcessor(JNothing)
  val questionLine = "B\tSentence 1. Sentence 2 ___. (A) answer 1 (B) answer 2 (C) answer 3 (D) answer 4"
  val question = ScienceQuestion(
    Seq("Sentence 1.", "Sentence 2 ___."),
    Seq(
      Answer("answer 1", false),
      Answer("answer 2", true),
      Answer("answer 3", false),
      Answer("answer 4", false)
    )
  )
  val questionWithWhPhrase = ScienceQuestion(
    Seq("Which option is the answer?"),
    Seq(
      Answer("true", false),
      Answer("false", true)
    )
  )


  "parseQuestionLine" should "correctly split the question and answer, and the answer options" in {
    processor.parseQuestionLine(questionLine) should be(question)
  }

  "fillInAnswerOptions" should "fill in blanks" in {
    processor.fillInAnswerOptions(question) should be(("Sentence 1. Sentence 2 ___.",
      Seq(
        ("Sentence 2 answer 1.", false),
        ("Sentence 2 answer 2.", true),
        ("Sentence 2 answer 3.", false),
        ("Sentence 2 answer 4.", false)
      )
    ))
  }

  it should "replace wh-phrases" in {
    processor.fillInAnswerOptions(questionWithWhPhrase) should be(("Which option is the answer?",
      Seq(
        ("true is the answer.", false),
        ("false is the answer.", true)
      )
    ))
  }
}
