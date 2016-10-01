package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.deep_qa.parse.DependencyTree
import org.allenai.deep_qa.parse.QuestionInterpreter
import org.allenai.deep_qa.parse.StanfordParser
import org.allenai.deep_qa.parse.Token
import org.allenai.deep_qa.parse.transformers

class QuestionInterpreterStep(
  val params: JValue,
  val fileUtil: FileUtil
) extends Step(Some(params), fileUtil) with SentenceProducer {
  implicit val formats = DefaultFormats
  override val name = "Question Interpreter"

  val validParams = baseParams ++ Seq(
    "question file",
    "interpreter",
    "output file"
  )
  JsonHelper.ensureNoExtras(params, name, validParams)

  val questionFile = (params \ "question file").extract[String]
  override val outputFile = (params \ "output file").extract[String]

  override val inputs: Set[(String, Option[Step])] = Set((questionFile, None))
  override val outputs = Set(outputFile)
  override val paramFile = outputFile.dropRight(4) + "_params.json"
  override val inProgressFile = outputFile.dropRight(4) + "_in_progress"

  val questionInterpreter = QuestionInterpreter.create(params \ "interpreter")

  override def _runStep() {
    val rawQuestions = fileUtil.readLinesFromFile(questionFile).par
    val outputLines = rawQuestions.flatMap(questionLine => {
      try {
        val processedLines = questionInterpreter.processQuestion(questionLine)
        if (processedLines.size != 1 && processedLines.size != 4) {
          // We'll punt on handling this for now, as some downstream code relies on having 4
          // options for each question. TODO(matt)
          Seq()
        } else {
          processedLines
        }
      } catch {
        case e: NoSuchElementException => { logger.error(s"Error processing $questionLine"); Seq() }
      }
    }).seq
    outputSentences(outputLines)
  }
}
