package org.allenai.semparse.pipeline.science_data

import org.scalatest._

import org.json4s._
import org.json4s.JsonDSL._

import com.mattg.util.FileUtil

class ScienceQuestionProcessorSpec extends FlatSpecLike with Matchers {
  val params: JValue = ("question file" -> "/dev/null") ~ ("data name" -> "test data")
  val processor = new ScienceQuestionProcessor(params, new FileUtil)


  "parseQuestionLine" should "correctly split the question and answer, and the answer options" in {
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
    processor.parseQuestionLine(questionLine) should be(question)
  }

  "fillInAnswerOptions" should "fill in blanks" in {
    val question = ScienceQuestion(
      Seq("Sentence 1.", "Sentence 2 ___."),
      Seq(
        Answer("answer 1", false),
        Answer("answer 2", true),
        Answer("answer 3", false),
        Answer("answer 4", false)
      )
    )
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
    val question = ScienceQuestion(
      Seq("Which option is the answer?"),
      Seq(
        Answer("true", false),
        Answer("false", true)
      )
    )
    processor.fillInAnswerOptions(question) should be(Some("Which option is the answer?",
      Seq(
        ("True is the answer.", false),
        ("False is the answer.", true)
      )
    ))
  }

  it should "handle \"where is\" questions" in {
    val question = ScienceQuestion(
      Seq("Where is the answer?"),
      Seq(
        Answer("the ground", false),
        Answer("the sky", true)
      )
    )
    processor.fillInAnswerOptions(question) should be(Some("Where is the answer?",
      Seq(
        ("The answer is in the ground.", false),
        ("The answer is in the sky.", true)
      )
    ))
  }

  it should "handle \"where is\" questions with wh-movement" in {
    val questionText = "Where is most of Earth's water located?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("oceans", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("Most of Earth's water is located in oceans", true))))
  }

  it should "not add \"in\" if the answer already has it" in {
    val question = ScienceQuestion(
      Seq("Where is the answer?"),
      Seq(
        Answer("in the ground", false),
        Answer("in the sky", true)
      )
    )
    processor.fillInAnswerOptions(question) should be(Some("Where is the answer?",
      Seq(
        ("The answer is in the ground.", false),
        ("The answer is in the sky.", true)
      )
    ))
  }

  it should "handle \"how are\" questions" in {
    val question = ScienceQuestion(
      Seq("How are genes usually grouped inside a cell?"),
      Seq(
        Answer("By themselves", false),
        Answer("In pairs", true),
        Answer("In fours", false),
        Answer("In threes", false)
      )
    )
    processor.fillInAnswerOptions(question) should be(Some("How are genes usually grouped inside a cell?",
      Seq(
        ("Genes are usually grouped inside a cell by themselves.", false),
        ("Genes are usually grouped inside a cell in pairs.", true),
        ("Genes are usually grouped inside a cell in fours.", false),
        ("Genes are usually grouped inside a cell in threes.", false)
      )
    ))
  }

  it should "undo wh-movement" in {
    val questionText = "What is the male part of a flower called?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("Stamen", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("The male part of a flower is called stamen.", true))))
  }

  it should "undo nested wh-movement" in {
    val questionText = "From which part of the plant does a bee get food?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("flower", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("A bee gets food from flower.", true))))
  }

  it should "handle another wh-question" in {
    val questionText = "What is the end stage in a human's life cycle?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("death", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("The end stage in a human's life cycle is death.", true))))
  }
}
