package org.allenai.dlfa.parse

import com.typesafe.scalalogging.LazyLogging

import org.json4s._

/**
 * WhMovers undo wh-movement, converting a wh-question into a declarative sentence involving a
 * blank ("___").  For example, it might change "Which of the following is best?" to "___ is best.",
 * or "What can you use to do this?" to "You can use ___ to do this."
 *
 * The idea is that we can then trivially fill in the blank with other words (like answer options).
 * Our current code for doing this, though, isn't great, so if you can devise a model that doesn't
 * need to rely on this, it will be less brittle.
 */
abstract class WhMover extends LazyLogging {
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

/**
 * An attempt at undoing wh-movement, written by me (Matt Gardner).  It kind of works, sometimes.
 * You can see some of the cases where it works in QuestionInterpreterSpec (TODO(matt): move some
 * of those tests into a WhMoverSpec, where they belong).
 *
 * One of the huge problems with this is that it relies on a correct dependency parse of the
 * sentence, and the Stanford parser doesn't do a great job on questions.
 */
class MattsWhMover(parser: Parser) extends WhMover {

  def whQuestionToFillInTheBlank(question: ScienceQuestion, lastSentence: String): Option[String] = {
    val parse = parser.parseSentence(lastSentence)
    parse.dependencyTree match {
      case None => { logger.error(s"couldn't parse question: $question"); return None }
      case Some(tree) => {
        // These lines are for development; I uncomment them when I'm trying to fix a bug.
        // TODO(matt): I really should change DependencyTree.print() to a method that returns a
        // string, and call logger.debug() on these statements.
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

/**
 * This is supposed to be the wh-movement code in the AI2 codebase, written by Mark Hopkins.  The
 * trouble is that the code isn't public, so I can't include it in here yet.  These lines are
 * commented out until such time as that code is made public.
 */
class MarksWhMover extends WhMover {
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
