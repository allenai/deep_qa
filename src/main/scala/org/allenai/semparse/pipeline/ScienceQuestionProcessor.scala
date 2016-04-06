package org.allenai.semparse.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.semparse.parse.DependencyTree
import org.allenai.semparse.parse.LogicalFormGenerator
import org.allenai.semparse.parse.Predicate
import org.allenai.semparse.parse.StanfordParser

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

  override def _runStep() {
    val rawQuestions = fileUtil.readLinesFromFile(questionFile).par
    val splitQuestions = rawQuestions.map(splitAnswerFromQuestion)
  }

  /**
   *  Takes a question line formatted as "[correct_answer]\t[question] [answers]" as splits it out
   *  into (question string, correct answer string, set of incorrect answer strings).
   */
  def splitAnswerFromQuestion(questionLine: String): (String, String, Set[String]) = {
  }
}
