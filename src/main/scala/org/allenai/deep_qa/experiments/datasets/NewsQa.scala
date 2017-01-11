package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for NewsQA data files.
 * Before using this, you must run the script for cleaning up NewsQA
 * in deep_qa/src/main/python/scripts/clean_newsqa.py.
 */
object NewsQaDatasets {

  def newsQaFile(newsQaDir: String, split: String): JValue = {
    val outputDirectory = newsQaDir + "processed/"
    val inputFile = newsQaDir + "cleaned/" + s"${split}.csv.clean"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "newsQa") ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def newsQaDataset(newsQaDir: String, split: String): JValue = {
    val file = newsQaFile(newsQaDir, split)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/news_qa/split_data/"

  val trainFile = newsQaFile(baseDir, "train")
  val trainDataset = newsQaDataset(baseDir, "train")
  val devFile = newsQaFile(baseDir, "dev")
  val devDataset = newsQaDataset(baseDir, "dev")
}
