package org.allenai.dlfa.parse

import com.typesafe.scalalogging.LazyLogging

import org.json4s._

abstract class WhMover(params: JValue) extends LazyLogging {
  def whQuestionToFillInTheBlank(question: ScienceQuestion, lastSentence: String): Option[String]
}

object WhMover {
  def create(params: JValue): WhMover = {
    params match {
      case JString("matt's") => new MattsWhMover(new StanfordParser)
      case JString("mark's") => new MarksWhMover
      case _ => throw new IllegalStateException("unrecognized wh-mover parameters")
    }
  }
}

class MattsWhMover(parser: Parser) extends WhMover(JNothing) {

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
