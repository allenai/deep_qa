package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This is a collection of DatasetStep specifications, for easy re-use in various experiments.
 */
object SnliDatasets {

  def snliDataset(snliDir: String, split: String): JValue = {
    val outputDirectory = snliDir + "processed/"
    val inputFile = snliDir + s"snli_1.0_${split}.txt"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("data files" -> List(
      ("sentence producer type" -> "dataset reader") ~
      ("reader" -> "snli") ~
      ("input file" -> inputFile) ~
      ("output files" -> outputFiles)))
  }

  val baseDir = "/efs/data/dlfa/snli/"

  val train = snliDataset(baseDir, "train")
  val dev = snliDataset(baseDir, "dev")
  val test = snliDataset(baseDir, "test")
}
