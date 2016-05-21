package org.allenai.semparse.pipeline.science_data

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
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Science Question Processor"

  val validParams = Seq("question file", "data name")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val questionFile = (params \ "question file").extract[String]
  val dataName = (params \ "data name").extract[String]
  val outputFile = s"data/science/$dataName/processed_questions.txt"

  override val inputs: Set[(String, Option[Step])] = Set((questionFile, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.replace(".txt", "_params.json")
  override val inProgressFile = outputFile.replace(".txt", "_in_progress")

  val parser = new StanfordParser
  val logicalFormGenerator = new LogicalFormGenerator(JNothing)

  override def _runStep() {
    val rawQuestions = fileUtil.readLinesFromFile(questionFile).par
    val questions = rawQuestions.map(parseQuestionLine)
    val filledInAnswers = questions.map(fillInAnswerOptions)
    val outputLines = filledInAnswers.seq.flatMap(_ match {
      case None => Seq()
      case Some(filledInAnswer) => {
        Seq(filledInAnswer._1) ++ filledInAnswer._2.map(answerSentence => {
          val (text, correct) = answerSentence
          val parsedAnswerOption = parser.parseSentence(text)
          val logicalForm = parsedAnswerOption.dependencyTree match {
            case None => None
            case Some(tree) => logicalFormGenerator.getLogicalForm(tree)
          }
          val logicalFormStr = logicalForm match {
            case None => ""
            case Some(logicalForm) => logicalForm.toLisp
          }
          val correctString = if (correct) "1" else "0"
          s"$text\t$logicalFormStr\t$correctString"
        }) ++ Seq("")
      }
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
   * answer filled in)).  It's an option, because we're going to explicitly punt on some questions
   * for now.
   *
   * Currently, this just takes the _last sentence_ and tries to fill in the blank (or replace the
   * wh-phrase).  This will have to change once my model can incorporate the whole question text.
   */
  def fillInAnswerOptions(question: ScienceQuestion): Option[(String, Seq[(String, Boolean)])] = {
    val questionText = question.sentences.mkString(" ")
    val lastSentence = question.sentences.last

    // I don't want to deal with these questions right now, and the current model of semantics
    // can't deal with it, anyway.
    if (lastSentence.contains("How many")) return None

    val sentenceWithBlank = if (lastSentence.contains("___")) {
      lastSentence
    } else {
      val parse = parser.parseSentence(lastSentence)
      parse.dependencyTree match {
        case None => { System.err.println(s"couldn't parse question: $question"); return None }
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
              System.err.println(s"exception finding wh-phrase: $question")
              return None
            }
          }
          whPhrase match {
            case None => {
              System.err.println(s"question didn't have blank or wh-phrase: $question")
              movementUndone.print()
              return None
            }
            case Some(whTree) => {
              if (whTree.lemmaYield == "where" && question.answers.exists(!_.text.startsWith("in "))) {
                // Here we're going to special case "where" questions, because replacing "where"
                // with the answer requires the insertion of "in", unless the answer already starts
                // with "in".
                val movementUndoneSentence = movementUndone._yield
                // I should be able to do this with a regex, but I couldn't get it to work
                // immediately, and this works.
                val toReplace = if (movementUndoneSentence.contains("Where")) "Where" else "where"
                movementUndoneSentence.replace(toReplace, "in ___.")
              } else {
                val index = whTree.tokens.map(_.index).min
                val newTree = DependencyTree(Token("___", "NN", "___", index), Seq())
                transformers.replaceTree(movementUndone, whTree, newTree)._yield + "."
              }
            }
          }
        }
      }
    }
    val answerSentences = question.answers.map(a => (sentenceWithBlank.replace("___", a.text.toLowerCase), a.isCorrect))
    Some((questionText, answerSentences.map(makeAnswerUpperCase)))
  }

  def makeAnswerUpperCase(answer: (String, Boolean)): (String, Boolean) = {
    (answer._1.capitalize, answer._2)
  }
}
