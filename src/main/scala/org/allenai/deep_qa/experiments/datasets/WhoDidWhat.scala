package org.allenai.deep_qa.experiments.datasets

import org.json4s._
import org.json4s.JsonDSL._

/**
 * This object contains a bunch of JValue specifications for WDW data files.
 */
object WhoDidWhatDatasets {

  def WhoDidWhatFile(wdwDir: String, split: String, relaxed: Boolean = false): JValue = {
    val suppression = if (relaxed){
      if (split != "train"){
        // relaxed is for training only
        throw new IllegalArgumentException("Relaxed dataset can only be used for train.")
      }
      "Relaxed/"
    } else {
      "Strict/"
    }

    val outputDirectory = wdwDir + "processed/" + suppression.toLowerCase
    val inputFile = wdwDir + suppression + s"${split}.xml"
    val outputFiles = Seq(outputDirectory + s"${split}.tsv")
    ("sentence producer type" -> "dataset reader") ~
    ("reader" -> "who did what") ~
    ("input file" -> inputFile) ~
    ("output files" -> outputFiles)
  }

  def WhoDidWhatDataset(wdwDir: String, split: String, relaxed: Boolean = false): JValue = {
    val file = WhoDidWhatFile(wdwDir, split, relaxed)
    ("data files" -> List(file))
  }

  val baseDir = "/efs/data/dlfa/who_did_what/"

  val trainFile = WhoDidWhatFile(baseDir, "train")
  val train = WhoDidWhatDataset(baseDir, "train")
  val devFile = WhoDidWhatFile(baseDir, "dev")
  val dev = WhoDidWhatDataset(baseDir, "dev")
  val relaxedTrainFile = WhoDidWhatFile(baseDir, "train", true)
  val relaxedTrain = WhoDidWhatDataset(baseDir, "train", true)
}
