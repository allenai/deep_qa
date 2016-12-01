package org.allenai.deep_qa.pipeline

import org.allenai.deep_qa.experiments.datasets.BabiDatasets

import com.mattg.pipeline.Step
import com.mattg.util.FileUtil
import org.json4s._
import org.json4s.JsonDSL._

import scala.collection.mutable

/**
 * This Step runs a set of models on all of the bAbI tasks, and shows the results in a table.
 *
 * models: a list of (model name, model params) pairs.  We'll add the dataset parameters here, so
 * leave those out.
 */
class BabiEvaluator(
  evaluatorName: Option[String] = None,
  models: Seq[(String, JValue)],
  fileUtil: FileUtil
) extends Step(None, fileUtil) {

  def addBabiParams(modelParams: JValue, modelName: String, taskNumber: Int): JValue = {
    ("model params" -> modelParams) ~
    ("name" -> s"${modelName}_task_${taskNumber}") ~
    ("dataset" -> BabiDatasets.task(taskNumber))
  }

  val modelTrainers = models.map { case (name, params) => {
    // 8 and 19 have multiple outputs, and we haven't dealt with that yet...
    val tasks = (1 to 20).filterNot(Set(8, 19).contains).map(i => addBabiParams(params, name, i))
    tasks.map(new NeuralNetworkTrainerStep(_, fileUtil))
  }}
  val trainers = modelTrainers.flatten
  val evaluationHash = trainers.map(_.modelHash).hashCode().toHexString
  override val name = evaluatorName.getOrElse(evaluationHash)
  val modelHashFile = s"/efs/data/dlfa/models/evaluations/${name}_${evaluationHash}.txt"

  override val inputs: Set[(String, Option[Step])] = trainers.map(trainer => (trainer.logFile, Some(trainer))).toSet
  override val outputs: Set[String] = Set()
  override val inProgressFile = modelHashFile.dropRight(4) + "_in_progress"

  logger.info(s"Writing all model hashes used for experiment $name to $modelHashFile")
  fileUtil.writeLinesToFile(modelHashFile, trainers.map(_.modelHash))

  override def _runStep() {
    val metric = "best validation accuracy"
    val resultsDict = modelTrainers.zip(models).map(entry => {
      val (tasks, (modelName, _)) = entry
      val taskResults = tasks.par.map(task => {
        val stats = readStatsFromFile(task.logFile)
        val taskNumber = task.modelName.split("task_")(1).toInt
        (taskNumber -> stats(metric))
      }).seq.toMap
      (modelName -> taskResults)
    }).toMap
    val metrics = Seq("val_acc")
    println("Methods:")
    for (((modelName, _), index) <- models.zipWithIndex) {
      println(s"${index+1}: ${modelName}")
    }
    println()
    val taskHeader = "Task"
    print(f"$taskHeader%-5s")
    for (methodNumber <- (1 to models.size)) {
      print(f"$methodNumber%10s")
    }
    println()
    val modelValues = new mutable.HashMap[String, mutable.ArrayBuffer[Double]]
    for (taskNumber <- (1 to 20)) {
      print(f"${taskNumber}%5d")
      for (((modelName, _), methodNumber) <- models.zipWithIndex) {
        val value = resultsDict.getOrElse(modelName, Map()).getOrElse(taskNumber, -1.0)
        if (value >= 0) {
          modelValues.getOrElseUpdate(modelName, new mutable.ArrayBuffer()) += value
          print(f"${value}%10.4f")
        } else {
          val message = "No value"
          print(f"${message}%10s")
        }
      }
      println()
    }
    print("-----")
    for (_ <- (1 to models.size)) {
      print("----------")
    }
    println()
    print("Ave. ")
    for ((modelName, _) <- models) {
      val average = modelValues(modelName).sum / modelValues(modelName).size
        print(f"${average}%10.4f")
    }
    println()
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
