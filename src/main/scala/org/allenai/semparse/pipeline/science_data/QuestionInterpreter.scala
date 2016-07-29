package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

/*
import org.allenai.ari.controller.questionparser._
import org.allenai.ari.controller.decomposer.SimpleLongAnswerGenerator
import org.allenai.ari.models.ParentheticalChoiceIdentifier
import org.allenai.ari.models.SquareBracketedChoiceIdentifier
import org.allenai.nlpstack.parse.poly.polyparser.Parser
import org.allenai.nlpstack.parse.poly.core.{Token => PPToken}
*/

import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.StanfordParser
import org.allenai.semparse.parse.Token
import org.allenai.semparse.parse.transformers

import com.typesafe.scalalogging.LazyLogging

case class Answer(text: String, isCorrect: Boolean)
case class ScienceQuestion(sentences: Seq[String], answers: Seq[Answer])

class QuestionInterpreter(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Question Interpreter"

  val validParams = baseParams ++ Seq("question file", "wh-movement", "output file")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val questionFile = (params \ "question file").extract[String]
  override val outputFile = (params \ "output file").extract[String]
  val whMover = WhMover.create(params \ "wh-movement")

  override val inputs: Set[(String, Option[Step])] = Set((questionFile, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"

  val parser = new StanfordParser

  override def _runStep() {
    val rawQuestions = fileUtil.readLinesFromFile(questionFile).par
    val outputLines = rawQuestions.flatMap(questionLine => {
      val question = parseQuestionLine(questionLine)
      val sentenceStrings = fillInAnswerOptions(question)
      sentenceStrings match {
        case None => Seq()
        case Some(filledInAnswers) => {
          val answerStrings = filledInAnswers.map(answerSentence => {
            val (text, correct) = answerSentence
            val correctString = if (correct) "1" else "0"
            s"$text\t$correctString"
          })
          // TODO(matt): figure out output formats.  You probably want some options here, so that
          // you have a version that retains the original question for visual inspection, but also
          // a version that doesn't keep around the original question, for easier processing later
          // in the pipeline.
          //Seq(question.sentences.mkString(" ")) ++ answerStrings ++ Seq("")
          answerStrings
        }
      }
    }).seq
    outputSentences(outputLines)
  }

  /**
   *  Parses a question line formatted as "[correct_answer]\t[question] [answers]", returning a
   *  ScienceQuestion object.
   *
   *  TODO(matt): There's code in the question decomposer that will handle this more generally, but
   *  that code is heavy...
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
   * answer filled in)).  It's an option, because we're going to explicitly punt on some questions
   * for now.
   *
   * Currently, this just takes the _last sentence_ and tries to fill in the blank (or replace the
   * wh-phrase).  This will have to change once my model can incorporate the whole question text.
   */
  def fillInAnswerOptions(question: ScienceQuestion): Option[Seq[(String, Boolean)]] = {
    val lastSentence = question.sentences.last

    val sentenceWithBlank = if (lastSentence.contains("___")) {
      lastSentence
    } else {
      whMover.whQuestionToFillInTheBlank(question, lastSentence) match {
        case None => return None
        case Some(sentence) => sentence
      }
    }
    val answerSentences = question.answers.map(a => (sentenceWithBlank.replace("___", a.text.toLowerCase), a.isCorrect))
    Some(answerSentences.map(makeAnswerUpperCase))
  }

  def makeAnswerUpperCase(answer: (String, Boolean)): (String, Boolean) = {
    (answer._1.capitalize, answer._2)
  }
}

abstract class WhMover(params: JValue) extends LazyLogging {
  def whQuestionToFillInTheBlank(question: ScienceQuestion, lastSentence: String): Option[String]
}

object WhMover {
  def create(params: JValue): WhMover = {
    params match {
      case JString("matt's") => new MattsWhMover
      case JString("mark's") => new MarksWhMover
      case _ => throw new IllegalStateException("unrecognized wh-mover parameters")
    }
  }
}

class MattsWhMover extends WhMover(JNothing) {

  val parser = new StanfordParser

  def whQuestionToFillInTheBlank(question: ScienceQuestion, lastSentence: String): Option[String] = {
    val parse = parser.parseSentence(lastSentence)
    parse.dependencyTree match {
      case None => { logger.error(s"couldn't parse question: $question"); return None }
      case Some(tree) => {
        //println("Original tree:")
        //tree.print()
        val fixedCopula = transformers.MakeCopulaHead.transform(tree)
        //println("Fixed copula tree:")
        //fixedCopula.print()
        val movementUndone = transformers.UndoWhMovement.transform(fixedCopula)
        //println("Moved tree:")
        //movementUndone.print()
        val whPhrase = try {
          transformers.findWhPhrase(movementUndone)
        } catch {
          case e: Throwable => {
            logger.error(s"exception finding wh-phrase: $question")
            return None
          }
        }
        whPhrase match {
          case None => {
            logger.error(s"question didn't have blank or wh-phrase: $question")
            movementUndone.print()
            return None
          }
          case Some(whTree) => {
            if (isWhereQuestion(whTree, movementUndone, question)) {
              Some(fillInWhereQuestion(movementUndone, question))
            } else if (isWhyQuestion(whTree, movementUndone, question)) {
              Some(fillInWhyQuestion(movementUndone, question))
            } else {
              val index = whTree.tokens.map(_.index).min
              val newTree = DependencyTree(Token("___", "NN", "___", index), Seq())
              Some(transformers.replaceTree(movementUndone, whTree, newTree)._yield + ".")
            }
          }
        }
      }
    }
  }

  def isWhereQuestion(
    whTree: DependencyTree,
    finalTree: DependencyTree,
    question: ScienceQuestion
  ): Boolean = {
    // We check to see if the question is "where", _and_ there is not already an "in" in the answer
    // option.  If there is already an "in", we'll just handle this like normal.
    whTree.lemmaYield == "where" &&
    !finalTree.children.exists(_._2 == "prep") &&
    !question.answers.exists(_.text.toLowerCase.startsWith("in ")) &&
    !question.answers.exists(_.text.toLowerCase.startsWith("from "))
  }

  def fillInWhereQuestion(finalTree: DependencyTree, question: ScienceQuestion): String = {
    val finalSentence = finalTree._yield
    val toReplace = if (finalSentence.contains("Where")) "Where" else "where"
    finalSentence.replace(toReplace, "in ___.")
  }

  def isWhyQuestion(
    whTree: DependencyTree,
    finalTree: DependencyTree,
    question: ScienceQuestion
  ): Boolean = {
    // Similar to isWhereQuestion
    whTree.lemmaYield == "why" && question.answers.exists(!_.text.startsWith("because "))
  }

  def fillInWhyQuestion(finalTree: DependencyTree, question: ScienceQuestion): String = {
    val finalSentence = finalTree._yield
    val toReplace = if (finalSentence.contains("Why")) "Why" else "why"
    finalSentence.replace(toReplace, "because ___.")
  }
}

class MarksWhMover extends WhMover(JNothing) {
  //val fillInTheBlankGenerator = FillInTheBlankGenerator.mostRecent
  //val fillInTheBlankProcessor = new FillInTheBlankQuestionProcessor
  override def whQuestionToFillInTheBlank(question: ScienceQuestion, lastSentence: String): Option[String] = {
    /*
    logger.info(s"last sentence: $lastSentence")
    val standardQuestion = StandardQuestion(lastSentence)
    val fillInTheBlankQuestion = fillInTheBlankGenerator.generateFITB(standardQuestion).get
    logger.info(s"fill in the blank question: ${fillInTheBlankQuestion}")
    val finalQuestion = fillInTheBlankQuestion.text.replace("BLANK_", "___").replaceAll(" ,", ",").replace(" .", ".")
    logger.info(finalQuestion)
    Some(finalQuestion)
    */
    None
  }
}
