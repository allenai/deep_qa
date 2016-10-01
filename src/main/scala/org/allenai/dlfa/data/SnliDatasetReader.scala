package org.allenai.dlfa.data

import scala.collection.mutable

import com.mattg.util.FileUtil

class SnliDatasetReader(fileUtil: FileUtil) {
  def readFile(filename: String): Dataset[SnliInstance] = {
    val instances = fileUtil.flatMapLinesFromFile(filename, line => {
      val fields = line.split("\t")
      val label = fields(0) match {
        case "neutral" => Some("neutral")
        case "entailment" => Some("entails")
        case "contradiction" => Some("contradicts")
        case _ => None
      }
      val text = fields(5)
      val hypothesis = fields(6)
      label.map(l => SnliInstance(text, hypothesis, Some(l))).toSeq
    })
    Dataset(instances)
  }
}

// TODO(matt): Move this somewhere else, or just remove it, once this is integrated as a Step in
// the pipeline.
object SnliDatasetReader {
  def main(args: Array[String]) {
    val fileUtil = new FileUtil
    val reader = new SnliDatasetReader(fileUtil)
    val dataset = reader.readFile("/home/mattg/data/snli/snli_1.0_train.txt")
    dataset.writeToFiles(
      Seq("/home/mattg/data/snli/train.tsv"),
      false,
      fileUtil
    )
  }
}
