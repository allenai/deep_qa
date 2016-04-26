package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import org.allenai.semparse.pipeline.base.Trainer

class ScienceQuestionScorer(
  params: JValue,
  fileUtil: FileUtil
) extends Step(Some(params), fileUtil) {
  implicit val formats = DefaultFormats
  override val name = "Science Question Scorer"

  val validParams = Seq("questions", "model")
  JsonHelper.ensureNoExtras(params, name, validParams)

  val questionProcessor = new ScienceQuestionProcessor(params \ "questions", fileUtil)
  val trainer = new Trainer(params \ "model", fileUtil)
  val trainingDataProcessor = trainer.processor
  val modelType = trainer.modelType
  val questionDataName = questionProcessor.dataName
  val trainingDataName = trainer.dataName

  val questionFile = questionProcessor.outputFile

  val lispBase = "src/main/lisp"

  // These are code files that are common to all models, and just specify the base lisp
  // environment.
  val baseEnvFile = s"$lispBase/environment.lisp"
  val uschemaEnvFile = s"$lispBase/uschema_environment.lisp"

  // These are code files, specifying the model we're using.
  val lispModelFile = s"$lispBase/model_${modelType}.lisp"
  val evalLispFile = s"$lispBase/eval_uschema.lisp"

  val handwrittenLispFiles = Seq(baseEnvFile, uschemaEnvFile, lispModelFile, evalLispFile)

  // This one is a (hand-written) parameter file that will be passed to lisp code.
  val sfeSpecFile = trainer.sfeSpecFile

  // These are data files, produced by TrainingDataProcessor.
  val entityFile = s"data/${trainingDataName}/entities.lisp"
  val wordsFile = s"data/${trainingDataName}/words.lisp"

  val dataFiles = Seq(entityFile, wordsFile)

  val serializedModelFile = trainer.serializedModelFile

  val inputFiles = dataFiles ++ handwrittenLispFiles
  val extraArgs = Seq(sfeSpecFile, trainingDataName, serializedModelFile)

  val outputFile = s"results/science/$questionDataName/$trainingDataName/output.txt"

  // At this point we're finally ready to override the Step methods.
  override val paramFile = outputFile.replace("output.txt", "params.json")
  override val inProgressFile = outputFile.replace("output.txt", "in_progress")
  override val inputs =
    handwrittenLispFiles.map((_, None)).toSet ++
    Set((sfeSpecFile, None), (questionFile, None)) ++
    dataFiles.map((_, Some(trainingDataProcessor))).toSet ++
    Set((serializedModelFile, Some(trainer)))
  override val outputs = Set(outputFile)

  val questionEvaluationFormat = "(expression-eval (quote (get-predicate-marginals (lambda (x) %s (array \"dummy\")))))"

  def _runStep() {
    // Here we just need to iterate over the questions in the question file, score each of the
    // options, and output a file with the scores.
    val questions = readQuestionFile()

    val scores = questions.par.map(question => {
      val score = scoreQuestion(question)
      (question, score)
    })

    // TODO(matt): output scores.  Easiest would be to flatmap over the scored questions, similar
    // to how QuestionProcessor works.
  }

  def scoreQuestion(question: ProcessedQuestion): Double = {
    // TODO(matt)
    0.0
  }

  def readQuestionFile(): Seq[ProcessedQuestion] = {
    val questions = new collection.mutable.ListBuffer[ProcessedQuestion]()
    var questionText: Option[String] = None
    var options = new collection.mutable.ListBuffer[ProcessedOption]()
    for (line <- fileUtil.getLineIterator(questionFile)) {
      questionText match {
        case None => {
          questionText = Some(line)
        }
        case Some(text) => {
          if (line.isEmpty) {
            val question = ProcessedQuestion(text, options.toSeq)
            questionText = None
            options.clear()
          } else {
            options += processOptionFromLine(line)
          }
        }
      }
    }
    questions.toSeq
  }

  def processOptionFromLine(line: String): ProcessedOption = {
    val fields = line.split("\t")
    val text = fields(0)
    val logicalForm = fields(1)
    val correct = fields(2) == "1"
    ProcessedOption(text, logicalForm, correct)
  }
}

case class ProcessedOption(text: String, logicalForm: String, correct: Boolean)
case class ProcessedQuestion(text: String, options: Seq[ProcessedOption])
