package org.allenai.semparse.pipeline.science_data

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import com.mattg.util.JsonHelper

import org.json4s._

import com.jayantkrish.jklol.lisp.ConsValue

import org.allenai.semparse.pipeline.base.Trainer
import org.allenai.semparse.Environment

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
    Set((sfeSpecFile, None), (questionFile, Some(questionProcessor))) ++
    dataFiles.map((_, Some(trainingDataProcessor))).toSet ++
    Set((serializedModelFile, Some(trainer)))
  override val outputs = Set(outputFile)

  val questionEvaluationFormat = "(expression-eval (quote (get-predicate-marginals (lambda (x) %s) (array \"dummy\"))))"

  def _runStep() {
    // Here we just need to iterate over the questions in the question file, score each of the
    // answer options, and output a file with the scores.
    val questions = readQuestionFile()
    val env = new Environment(inputFiles, extraArgs)

    val scores = questions.par.map(question => {
      val score = scoreQuestion(question, env)
      (question, score)
    })

    val outputLines = scores.flatMap(questionAndScores => {
      val (question, scores) = questionAndScores
      Seq(question.text) ++ scores.map(answerAndScore => {
        val (answer, score) = answerAndScore
        val text = answer.text
        val logicalForm = answer.logicalForm
        val correctString = if (answer.correct) "1" else "0"
        s"$text\t$logicalForm\t$correctString\t$score"
      }) ++ Seq("")
    }).seq
    fileUtil.writeLinesToFile(outputFile, outputLines)
  }

  def scoreQuestion(question: ProcessedQuestion, env: Environment): Seq[(ProcessedAnswer, Double)] = {
    println(s"Scoring answer options for question ${question.text}")
    question.answers.map(a => (a, scoreAnswer(a, env)))
  }

  def scoreAnswer(answer: ProcessedAnswer, env: Environment): Double = {
    if (answer.logicalForm.isEmpty()) return 0.0
    val expression = questionEvaluationFormat.format(answer.logicalForm)
    val result = env.evaluateSExpression(expression).getValue()
    // By construction, this array should have exactly one item in it.
    val scoreObject = result.asInstanceOf[Array[Object]](0)
    val cons = scoreObject.asInstanceOf[ConsValue]
    val list = ConsValue.consListToList(cons, classOf[Object])
    val score = list.get(0).asInstanceOf[Double].toDouble
    score
  }

  def readQuestionFile(): Seq[ProcessedQuestion] = {
    val questions = new collection.mutable.ListBuffer[ProcessedQuestion]()
    var questionText: Option[String] = None
    var answers = new collection.mutable.ListBuffer[ProcessedAnswer]()
    for (line <- fileUtil.getLineIterator(questionFile)) {
      questionText match {
        case None => {
          questionText = Some(line)
        }
        case Some(text) => {
          if (line.isEmpty) {
            val question = ProcessedQuestion(text, answers.toSeq)
            questionText = None
            answers.clear()
          } else {
            answers += processOptionFromLine(line)
          }
        }
      }
    }
    questions.toSeq
  }

  def processOptionFromLine(line: String): ProcessedAnswer = {
    val fields = line.split("\t")
    val text = fields(0)
    val logicalForm = fields(1)
    val correct = fields(2) == "1"
    ProcessedAnswer(text, logicalForm, correct)
  }
}

case class ProcessedAnswer(text: String, logicalForm: String, correct: Boolean)
case class ProcessedQuestion(text: String, answers: Seq[ProcessedAnswer])
