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
        ("The answer is true.", false),
        ("The answer is false.", true)
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
      Some(questionText, Seq(("Most of Earth's water is located in oceans.", true))))
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

  ignore should "handle \"how are\" questions (Stanford parser gets this wrong)" in {
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

  ignore should "undo nested wh-movement (Stanford parser gets this wrong)" in {
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

  it should "handle agents correctly" in {
    val questionText = "What is a waste product excreted by lungs?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("carbon dioxide", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("A waste product excreted by lungs is carbon dioxide.", true))))
  }

  it should "undo movement correctly" in {
    val questionText = "Which property of air does a barometer measure?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("pressure", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("A barometer does measure pressue.", true))))
  }

  it should "handle nested wh-phrases" in {
    val questionText = "Where do plants get energy from to make food?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("the sun", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("Plants do get energy from the sun to make food.", true))))
  }

  it should "order adjuncts correctly" in {
    val questionText = "What is the source of energy required to begin photosynthesis?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("sunlight", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("The source of energy required to begin photosynthesis is sunlight.", true))))
  }

  it should "deal with extra words correctly" in {
    val questionText = "The digestion process begins in which of the following?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("mouth", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("The digestion process begins in mouth.", true))))
  }

  it should "deal with PP attachment on the wh word and a redundant word in the answer correctly" in {
    val questionText = "Where is the pigment chrolophyll found in plants?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("in the leaves", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("The pigment chlorophyll is found in the leaves in plants.", true))))
  }

  it should "deal with does in question correctly" in {
    val questionText = "From which part of the plant does a bee get food?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("flower", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("A bee does get food from flower part of the plant.", true))))
  }

  it should "identify the object correctly in" in {
    val questionText = "How are genes usually grouped in a cell?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("in pairs", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("Genes are usually grouped in a cell in pairs.", true))))
  }

  it should "deal with wh questions with prepositions in answers correctly" in {
    val questionText = "Where do offspring get their genes?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("From their parents", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("Offspring do get their genes from their parents.", true))))
  }

  it should "deal with modal verbs correctly" in {
    val questionText = "What tool should this student use?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("magnet", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("This student should use magnet.", true))))
  }

  it should "get the word order right" in {
    val questionText = "Which of the following resources does an animal use for energy?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("food", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("An animal does use food for energy.", true))))
  }

  it should "deal with Why correctly" in {
    val questionText = "Why is competition important?"
    val question = ScienceQuestion(Seq(questionText), Seq(Answer("It maintains a natural balance", true)))
    processor.fillInAnswerOptions(question) should be(
      Some(questionText, Seq(("Competition is important because it maintains a natural balance.", true))))
  }
  
}
