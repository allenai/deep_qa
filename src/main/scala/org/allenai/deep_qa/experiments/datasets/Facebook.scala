package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This is a collection of DatasetStep specifications, for easy re-use in various experiments.
 */
object BabiDatasets {

  val taskNames = Map(
    1 -> "qa1_single-supporting-fact",
    2 -> "qa2_two-supporting-facts",
    3 -> "qa3_three-supporting-facts",
    4 -> "qa4_two-arg-relations",
    5 -> "qa5_three-arg-relations",
    6 -> "qa6_yes-no-questions",
    7 -> "qa7_counting",
    8 -> "qa8_lists-sets",
    9 -> "qa9_simple-negation",
    10 -> "qa10_indefinite-knowledge",
    11 -> "qa11_basic-coreference",
    12 -> "qa12_conjunction",
    13 -> "qa13_compound-coreference",
    14 -> "qa14_time-reasoning",
    15 -> "qa15_basic-deduction",
    16 -> "qa16_basic-induction",
    17 -> "qa17_positional-reasoning",
    18 -> "qa18_size-reasoning",
    19 -> "qa19_path-finding",
    20 -> "qa20_agents-motivations"
  )

  def babiDataset(babiDir: String, taskNumber: Int, train: Boolean): JValue = {
    val outputDirectory = babiDir + "processed/"
    val trainString = if (train) "train" else "test"
    val inputFile = babiDir + "en/" + taskNames(taskNumber) + "_" + trainString + ".txt"
    val outputFiles = Seq(
      outputDirectory + s"task_${taskNumber}_${trainString}.tsv",
      outputDirectory + s"task_${taskNumber}_${trainString}_background.tsv"
    )
    ("data files" -> List(
      ("sentence producer type" -> "dataset reader") ~
      ("reader" -> "babi") ~
      ("create sentence indices" -> true) ~
      ("input file" -> inputFile) ~
      ("output files" -> outputFiles)))
  }

  val baseDir = "/efs/data/dlfa/facebook/babi_v1.0/"

  val task1 = babiDataset(baseDir, 1, true)
  val task2 = babiDataset(baseDir, 2, true)
  val task3 = babiDataset(baseDir, 3, true)
  val task4 = babiDataset(baseDir, 4, true)
  val task5 = babiDataset(baseDir, 5, true)
  val task6 = babiDataset(baseDir, 6, true)
  val task7 = babiDataset(baseDir, 7, true)
  val task8 = babiDataset(baseDir, 8, true)
  val task9 = babiDataset(baseDir, 9, true)
  val task10 = babiDataset(baseDir, 10, true)
  val task11 = babiDataset(baseDir, 11, true)
  val task12 = babiDataset(baseDir, 12, true)
  val task13 = babiDataset(baseDir, 13, true)
  val task14 = babiDataset(baseDir, 14, true)
  val task15 = babiDataset(baseDir, 15, true)
  val task16 = babiDataset(baseDir, 16, true)
  val task17 = babiDataset(baseDir, 17, true)
  val task18 = babiDataset(baseDir, 18, true)
  val task19 = babiDataset(baseDir, 19, true)
  val task20 = babiDataset(baseDir, 20, true)

  def task(taskNumber: Int, train: Boolean=true): JValue = babiDataset(baseDir, taskNumber, train)

  val allTasks = Seq(task1, task2, task3, task4, task5, task6, task7, task8, task9,
    task10, task11, task12, task13, task14, task15, task16, task17, task18, task19, task20
  )
}

object ChildrensBookDatasets {

  def childrensDataset(dataDir: String, questionType: String, split: String): JValue = {
    val outputDirectory = dataDir + "processed/"
    val splitString = split match {
      case "train" => "train"
      case "test" => "test_2500ex"
      case "dev" => "valid_2500ex"
      case _ => throw new IllegalStateException("Invalid split: " + split)
    }
    val inputFile = dataDir + s"data/cbtest_${questionType}_${splitString}.txt"
    val outputFiles = Seq(
      outputDirectory + s"${questionType}_${split}.tsv",
      outputDirectory + s"${questionType}_${split}_background.tsv"
    )
    ("data files" -> List(
      ("sentence producer type" -> "dataset reader") ~
      ("reader" -> "children's books") ~
      ("create sentence indices" -> true) ~
      ("input file" -> inputFile) ~
      ("output files" -> outputFiles)))
  }

  val baseDir = "/efs/data/dlfa/facebook/childrens_books/"

  val commonNouns = childrensDataset(baseDir, "CN", "train")
  val namedEntities = childrensDataset(baseDir, "NE", "train")
  val prepositions = childrensDataset(baseDir, "P", "train")
  val verbs = childrensDataset(baseDir, "V", "train")

  val allTasks = Seq(commonNouns, namedEntities, prepositions, verbs)
}
