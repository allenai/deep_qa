package org.allenai.semparse.pipeline.science_data

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
  val questionWithWhere = ScienceQuestion(
    Seq("Where is the answer?"),
    Seq(
      Answer("the ground", false),
      Answer("the sky", true)
    )
  )
  val questionWithWhereAndIn = ScienceQuestion(
    Seq("Where is the answer?"),
    Seq(
      Answer("in the ground", false),
      Answer("in the sky", true)
    )
  )
  val questionWithHow = ScienceQuestion(
    Seq("How are genes usually grouped inside a cell?"),
    Seq(
      Answer("By themselves", false),
      Answer("In pairs", true),
      Answer("In fours", false),
      Answer("In threes", false)
    )
  )


  "parseQuestionLine" should "correctly split the question and answer, and the answer options" in {
    processor.parseQuestionLine(questionLine) should be(question)
  }

  "fillInAnswerOptions" should "fill in blanks" in {
    processor.fillInAnswerOptions(question) should be(Some("Sentence 1. Sentence 2 ___.",
      Seq(
        ("Sentence 2 answer 1.", false),
        ("Sentence 2 answer 2.", true),
        ("Sentence 2 answer 3.", false),
        ("Sentence 2 answer 4.", false)
      )
    ))
  }

  it should "replace wh-phrases" in {
    processor.fillInAnswerOptions(questionWithWhPhrase) should be(Some("Which option is the answer?",
      Seq(
        ("true is the answer.", false),
        ("false is the answer.", true)
      )
    ))
  }

  it should "handle \"where is\" questions" in {
    processor.fillInAnswerOptions(questionWithWhere) should be(Some("Where is the answer?",
      Seq(
        ("the answer is in the ground.", false),
        ("the answer is in the sky.", true)
      )
    ))
  }

  it should "not add \"in\" if the answer already has it" in {
    processor.fillInAnswerOptions(questionWithWhereAndIn) should be(Some("Where is the answer?",
      Seq(
        ("the answer is in the ground.", false),
        ("the answer is in the sky.", true)
      )
    ))
  }

  it should "handle \"how are\" questions" in {
    processor.fillInAnswerOptions(questionWithHow) should be(Some("How are genes usually grouped inside a cell?",
      Seq(
        ("genes are usually grouped inside a cell by themselves.", false),
        ("genes are usually grouped inside a cell in pairs.", true),
        ("genes are usually grouped inside a cell in fours.", false),
        ("genes are usually grouped inside a cell in threes.", false)
      )
    ))
  }
}
