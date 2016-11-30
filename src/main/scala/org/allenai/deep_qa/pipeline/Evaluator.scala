package org.allenai.deep_qa.pipeline

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import org.json4s._

import scala.collection.mutable

class Evaluator(models: Seq[JValue], fileUtil: FileUtil) extends Step(None, fileUtil) {
  override val name = "Evaluator"

  val trainers = models.map(new NeuralNetworkTrainerStep(_, fileUtil))
  val modelNames = trainers.map(_.modelName)
  val evaluatorName = modelNames.hashCode().toHexString

  override val inputs: Set[(String, Option[Step])] = trainers.map(trainer => (trainer.logFile, Some(trainer))).toSet
  override val outputs: Set[String] = Set()
  override val inProgressFile = s"./evaluator_${evaluatorName}_in_progress"

  override def _runStep() {
    val results = trainers.par.map(t => readStatsFromFile(t.logFile)).seq
    for ((trainer, result) <- trainers.zip(results)) {
      println(s"Results for model ${trainer.modelName} (hash: ${trainer.modelHash}):")
      for ((key, value) <- result) {
        println(s"${key}: ${value}")
      }
    }
  }

  def readStatsFromFile(logFile: String): Map[String, Double] = {
    println(s"Reading log file: $logFile")
    val validationAccuracies = fileUtil.flatMapLinesFromFile(logFile, line => {
      if (line.contains("val_acc: ")) {
        val fields = line.split("val_acc: ")
        Seq(fields(1).toDouble)
      } else {
        Seq()
      }
    })
    val bestValidationAccuracy = validationAccuracies.zipWithIndex.maxBy(_._1)
    val stats = new mutable.HashMap[String, Double]
    stats("best validation accuracy") = bestValidationAccuracy._1
    stats("best epoch") = bestValidationAccuracy._2
    stats.toMap
  }
}
