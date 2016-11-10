package org.allenai.deep_qa.parse

import com.mattg.util.JsonHelper

import org.json4s._

/*
 * TODO(matt): I would like to use AI2's internal question interpreter as an option here, but it's
 * not currently available publicly, so these imports are commented out.
import org.allenai.ari.controller.questionparser._
import org.allenai.ari.controller.decomposer.SimpleLongAnswerGenerator
import org.allenai.ari.models.ParentheticalChoiceIdentifier
import org.allenai.ari.models.SquareBracketedChoiceIdentifier
import org.allenai.nlpstack.parse.poly.polyparser.Parser
import org.allenai.nlpstack.parse.poly.core.{Token => PPToken}
*/

case class Answer(text: String, isCorrect: Boolean)
case class ScienceQuestion(sentences: Seq[String], answers: Seq[Answer])

/**
 * A QuestionInterpreter takes a question and converts it into a representation usable by other
 * parts of the system.
 *
 * In some instances, this means converting the question with its answer options into several
 * complete statements (e.g., "Cells contain ___. (A) DNA (B) HIV" would become
 * Seq("Cells contain DNA", "Cells contain HIV").  Sometimes this means trying to undo wh-movement,
 * which is hard and doesn't work very well.  In other cases it's just a light reformatting of the
 * original question text.
 *
 * In all cases, the input/output spec of a QuestionInterpreter is String => Seq(String), where the
 * input string is formatted as [correct option][tab][question text, including answer options].
 *
 * TODO(matt): it might make more sense to return an Instance here, instead of a Seq[String].
 */
abstract class QuestionInterpreter {

  val parser = new StanfordParser

  def processQuestion(questionLine: String): Seq[String] = {
    val question = parseQuestionLine(questionLine)
    interpretQuestion(question)
  }

  /**
   *  Parses a question line formatted as "[correct_answer]\t[question] [answers]", returning a
   *  ScienceQuestion object.
   *
   *  NOTE: this assumes the answer options always begin with 'A'  or '1' and go up from there, and
   *  are formatted like (A).  There's code in the Aristo question decomposer that will handle this
   *  a little more generally, but that code is heavy...
   */
  def parseQuestionLine(questionLine: String): ScienceQuestion = {
    val fields = questionLine.split("\t")

    val (correctAnswerString, rawQuestion) = if (fields.length == 2) {
      (Some(fields(0)), fields(1))
    } else {
      (None, fields(0))
    }

    // Are we dealing with 'A' as the first answer option, or '1'?
    val firstAnswerChar = if (rawQuestion.contains("(A)")) 'A' else '1'
    val correctAnswerOption = correctAnswerString.map(_.charAt(0) - firstAnswerChar)

    val firstAnswerString = s"($firstAnswerChar)"
    val (questionText, answerOptions) = rawQuestion.splitAt(rawQuestion.indexOf(firstAnswerString))
    val sentences = parser.splitSentences(questionText.trim)
    val options = answerOptions.trim.split("\\(.\\)")
    val answers = options.drop(1).zipWithIndex.map(optionWithIndex => {
      val option = optionWithIndex._1
      val index = optionWithIndex._2
      val text = option.trim
      val isCorrect = correctAnswerOption.map(index == _).getOrElse(false)
      Answer(text, isCorrect)
    }).toSeq
    ScienceQuestion(sentences, answers)
  }

  def interpretQuestion(question: ScienceQuestion): Seq[String]
}

object QuestionInterpreter {
  def create(params: JValue): QuestionInterpreter = {
    (params \ "type") match {
      case JString("fill in the blank") => new FillInTheBlankInterpreter(params)
      case JString("append answer") => new AppendAnswerInterpreter
      case JString("question and answer") => new QuestionAndAnswerInterpreter
      case _ => throw new IllegalStateException("unrecognized wh-mover parameters")
    }
  }
}

/**
 * The QuestionInterpreter is just a reformatting of the original question, returning a single
 * string that contains the question and all answer options.
 *
 * The format is [question text][tab]([answer option][answer separator])+[tab][correct answer index]
 *
 * The answer separator is currently "###".
 */
class QuestionAndAnswerInterpreter extends QuestionInterpreter {
  val answerSeparator = "###"

  override def interpretQuestion(question: ScienceQuestion): Seq[String] = {
    val questionText = question.sentences.mkString(" ")
    val answerText = question.answers.map(_.text).mkString(answerSeparator)
    val label = question.answers.indexWhere(_.isCorrect)
    Seq(s"$questionText\t$answerText\t$label")
  }
}


/**
 * QuestionToStatementInterpreters convert each answer option into separate statements.  The list
 * returned by processQuestion() will have length equal to the number of options in the question.
 */
abstract class QuestionToStatementInterpreter extends QuestionInterpreter {

  val baseParams = Seq("type")

  override def interpretQuestion(question: ScienceQuestion): Seq[String] = {
    val filledOptions = fillInAnswerOptions(question)
    filledOptions.map(_.map(answerSentence => {
      val (text, correct) = answerSentence
      val correctString = if (correct) "1" else "0"
      s"$text\t$correctString"
    })).toSeq.flatten.toSeq
  }

  def fillInAnswerOptions(question: ScienceQuestion): Option[Seq[(String, Boolean)]]
}

class FillInTheBlankInterpreter(params: JValue) extends QuestionToStatementInterpreter {

  val name = "Fill-in-the-blank Interpreter"
  val validParams = baseParams ++ Seq("wh-movement", "last sentence only")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val lastSentenceOnly = JsonHelper.extractWithDefault(params, "last sentence only", true)
  val whMover = WhMover.create(params \ "wh-movement")

  /**
   * Takes a question, finds the place where the answer option goes, and fills in that place with
   * all of the answer options.  The return value is (original question text, Seq(question with
   * answer filled in)).  It's an option, because we're going to explicitly punt on some questions
   * for now.
   *
   * Currently, this just takes the _last sentence_ and tries to fill in the blank (or replace the
   * wh-phrase).
   */
  override def fillInAnswerOptions(question: ScienceQuestion): Option[Seq[(String, Boolean)]] = {
    val lastSentence = question.sentences.last

    val sentenceWithBlank = if (lastSentence.contains("___")) {
      lastSentence
    } else {
      whMover.whQuestionToFillInTheBlank(question, lastSentence) match {
        case None => return None
        case Some(sentence) => sentence
      }
    }
    val answerSentences = question.answers.map(a => {
      val replacedLastSentence = sentenceWithBlank.replaceFirst("_+", a.text.toLowerCase)
      val answerSentence = if (lastSentenceOnly) {
        replacedLastSentence
      } else {
        question.sentences.dropRight(1).mkString(" ") + " " + replacedLastSentence
      }
      (answerSentence, a.isCorrect)
    })
    Some(answerSentences.map(makeAnswerUpperCase))
  }

  def makeAnswerUpperCase(answer: (String, Boolean)): (String, Boolean) = {
    (answer._1.capitalize, answer._2)
  }
}

class AppendAnswerInterpreter extends QuestionToStatementInterpreter {

  val name = "Append Answer Interpreter"

  val answerSeparator = " ||| "

  /**
   * Here we just append the answer after the question text, with a separator symbol in between.
   */
  override def fillInAnswerOptions(question: ScienceQuestion): Option[Seq[(String, Boolean)]] = {
    val lastSentence = question.sentences.last
    Some(question.answers.map(a => {
      val answerSentence = question.sentences.mkString(" ") + answerSeparator + a.text
      (answerSentence, a.isCorrect)
    }))
  }
}
