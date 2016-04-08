package org.allenai.semparse.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.LogicalFormGenerator
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.parse.StanfordParser
import org.allenai.semparse.parse.Token
import org.allenai.semparse.parse.transformers

case class Answer(text: String, isCorrect: Boolean)
case class ScienceQuestion(sentences: Seq[String], answers: Seq[Answer])

class ScienceQuestionProcessor(
  params: JValue,
  fileUtil: FileUtil = new FileUtil
) extends Step(Some(params), fileUtil) {
  override val name = "Science Question Processor"

  val validParams = Seq("question file", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val questionFile = JsonHelper.extractWithDefault(params, "question file", "data/science_questions.tsv")
  val outputFile = JsonHelper.extractWithDefault(params, "output file", "data/processed_science_questions.tsv")

  override val inputs: Set[(String, Option[Step])] = Set((questionFile, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputs.head.replace(".txt", "_params.json")

  val parser = new StanfordParser

  override def _runStep() {
    val rawQuestions = fileUtil.readLinesFromFile(questionFile).par
    val questions = rawQuestions.map(parseQuestionLine)
    val filledInAnswers = questions.map(fillInAnswerOptions)
    val outputLines = filledInAnswers.seq.flatMap(filledInAnswer => {
      Seq(filledInAnswer._1) ++ filledInAnswer._2.map(answerSentence => {
        val (text, correct) = answerSentence
        val correctString = if (correct) "1" else "0"
        s"$text\t$correctString"
      })
    })
    fileUtil.writeLinesToFile(outputFile, outputLines)
  }

  /**
   *  Parses a question line formatted as "[correct_answer]\t[question] [answers]", returning a
   *  ScienceQuestion object.
   */
  def parseQuestionLine(questionLine: String): ScienceQuestion = {
    val fields = questionLine.split("\t")
    val correctAnswerOption = fields(0).charAt(0) - 'A'
    val (questionText, answerOptions) = fields(1).splitAt(fields(1).indexOf("(A)"))
    val sentences = parser.splitSentences(questionText.trim)
    val options = answerOptions.trim.split("\\(.\\)")
    val answers = options.drop(1).zipWithIndex.map(optionWithIndex => {
      val option = optionWithIndex._1
      val index = optionWithIndex._2
      val text = option.trim
      val isCorrect = index == correctAnswerOption
      Answer(text, isCorrect)
    }).toSeq
    ScienceQuestion(sentences, answers)
  }

  /**
   * Takes a question, finds the place where the answer option goes, and fills in that place with
   * all of the answer options.  The return value is (original question text, Seq(question with
   * answer filled in)).
   *
   * Currently, this just takes the _last sentence_ and tries to fill in the blank (or replace the
   * wh-phrase).  This will have to change once my model can incorporate the whole question text.
   */
  def fillInAnswerOptions(question: ScienceQuestion): (String, Seq[(String, Boolean)]) = {
    val questionText = question.sentences.mkString(" ")
    val lastSentence = question.sentences.last
    val sentenceWithBlank = if (lastSentence.contains("___")) lastSentence else {
      val parse = parser.parseSentence(lastSentence)
      parse.dependencyTree match {
        case None => throw new RuntimeException(s"couldn't parse question: $question")
        case Some(tree) => {
          val whPhrase = transformers.findWhPhrase(tree)
          whPhrase match {
            case None => throw new RuntimeException(s"question didn't have blank or wh-phrase: $question")
            case Some(whTree) => {
              val index = whTree.tokens.map(_.index).min
              val newTree = DependencyTree(Token("___", "NN", "___", index), Seq())
              transformers.replaceTree(tree, whTree, newTree)._yield + "."
            }
          }
        }
      }
    }
    val answerSentences = question.answers.map(a => (sentenceWithBlank.replace("___", a.text), a.isCorrect))
    (questionText, answerSentences)
  }
}
